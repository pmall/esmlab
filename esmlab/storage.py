"""Persistence of computed masked-logits, keyed by ``(model, sequence, region)``.

Storage is a layer of its own: connectors compute logits and never persist
them, and this module persists logits and never computes them. Callers
orchestrate the two — ``has`` before compute, ``save`` after — so neither side
knows about the other. There is no decorator and no opt-out: logits are
expensive deterministic artifacts, so they are always kept.

Storage is a database. There is no filesystem backend, because a tree of
per-sequence directories is the wrong shape for tabular provenance and for
asking which models have scored a sequence.

Schema
------
One ``sequence_logits`` table, primary key ``(model, sequence_key)``.

- **The key is a digest, not the sequence.** Sequences run to thousands of
  residues, so the key column is
  ``sequence_key = blake2b(sequence, start, stop)`` and the sequence itself is a
  regular column that is never compared. The digest covers the sequence and the
  scored region, which keeps ``model`` a free-standing key column: one request
  has one key under every model, so ``WHERE sequence_key = ...`` answers which
  models have scored it. The same residues inside two different sequences are
  two rows, because the context differs and so do the logits. The *backend* is
  deliberately absent — logits for a given ``(model, sequence, region)`` are the
  same artifact whichever backend produced them, so a Modal run's results serve
  a later local run.
- **Logits are a raw C-order float32 blob** with ``n_positions`` / ``n_tokens``
  in their own columns, so a read is a reshape rather than a deserialization
  and :meth:`has` never transfers the blob at all. The vocab is JSON text.
- **The record is split in two.** ``label`` is a plain text column holding the
  display name, so ``WHERE label = 'stat1'`` works; ``metadata`` is a JSON
  column holding the arbitrary object the caller supplied (:mod:`esmlab.seqio`
  is where it comes from). Both engines can filter on it in SQL —
  ``json_extract`` in SQLite, ``metadata::jsonb`` in PostgreSQL — instead of
  decoding rows in Python.

Scope, writes, backends
-----------------------
Every implementation binds to one resource, and that resource is the only
scope: two stores pointed at the same database are the same store, and a store
holds exactly the entries some run put there. That is what makes a store the
unit a topic reports on - each topic keeps its own database, named by the
script that writes it, so a report covers that topic's runs and nothing else.
A ``stub`` run needs its own database too (``--sqlite-path
data/logits-stub.sqlite3``), or its fake logits will be served to a later real
run. No backend is special-cased in code.

Writes are a single upsert, which both engines speak, so a recompute replaces
the row and there is no state in which a later run sees a half-written entry
and skips it forever. The schema is created on open: a fresh database needs no
migration step.

:class:`LogitsStorage` is the Protocol; :class:`SqliteLogitsStorage` and
:class:`PostgresLogitsStorage` are the implementations, sharing everything but
paramstyle, blob type and connection setup via :class:`_SqlLogitsStorage`.
:func:`select_models` is where every consuming stage agrees on which of the
store's models one run covers.
"""

import argparse
import hashlib
import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import LiteralString, Protocol, cast

import numpy as np
import numpy.typing as npt

from esmlab.connectors.base import SequenceLogits
from esmlab.params import ParamSpec, ParamValue, resolve_params

type JsonValue = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)

# Whatever the caller wants to record about a sequence: a source, a list of
# targets, nested structure. Storage never inspects it, so its shape is the
# caller's to define and to change without a migration; the only constraint is
# that the top level is an object, so a reader never type-checks a bare scalar.
# A Mapping rather than a dict so a caller's narrower dict still fits, matching
# how SequenceLogits.vocab is typed.
type Metadata = Mapping[str, JsonValue]

TABLE: LiteralString = "sequence_logits"


def sqlite_params(default: Path) -> tuple[ParamSpec, ...]:
    """The SQLite flags, defaulting to ``default``.

    A function and not a constant because the default is the caller's: a store
    is the unit a topic reports on, so each script names the database its own
    topic writes and reads.
    """
    # A file path, not a credential: no env fallback. It stays a ParamSpec so
    # resolve_params still rejects it on a postgres run.
    return (
        ParamSpec(
            flag="--sqlite-path",
            dest="sqlite_path",
            env="",
            type=Path,
            default=default,
            help=(
                f"SQLite database file (default: {default}). "
                "Use a separate one for stub runs; the database is the only scope."
            ),
        ),
    )


POSTGRES_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec(
        flag="--postgres-host",
        dest="postgres_host",
        env="POSTGRES_HOST",
        default="localhost",
        help="PostgreSQL host (defaults to $POSTGRES_HOST, then localhost)",
    ),
    ParamSpec(
        flag="--postgres-port",
        dest="postgres_port",
        env="POSTGRES_PORT",
        type=int,
        default=5432,
        help="PostgreSQL port (defaults to $POSTGRES_PORT, then 5432)",
    ),
    ParamSpec(
        flag="--postgres-dbname",
        dest="postgres_dbname",
        env="POSTGRES_DB",
        default="esmlab",
        help="PostgreSQL database name (defaults to $POSTGRES_DB, then esmlab)",
    ),
    ParamSpec(
        flag="--postgres-user",
        dest="postgres_user",
        env="POSTGRES_USER",
        required=True,
        help="PostgreSQL role (defaults to $POSTGRES_USER from .env)",
    ),
    ParamSpec(
        flag="--postgres-password",
        dest="postgres_password",
        env="POSTGRES_PASSWORD",
        required=True,
        help="PostgreSQL password (defaults to $POSTGRES_PASSWORD from .env)",
    ),
)

STORAGES = ("sqlite", "postgres")


def storage_params(sqlite_default: Path) -> dict[str, tuple[ParamSpec, ...]]:
    """Every storage's parameters, by storage name, for one script's SQLite default."""
    return {"sqlite": sqlite_params(sqlite_default), "postgres": POSTGRES_PARAMS}


@dataclass(frozen=True)
class StoredLogits:
    """One persisted logits entry: the result plus its provenance.

    ``label`` is a display string carried alongside the entry (the sanitized
    FASTA header the sequence first arrived under). It is data, never identity:
    nothing looks an entry up by it, and two entries may share one.

    ``metadata`` is the structured half of the same record, an arbitrary JSON
    object supplied by the caller. It stays separate from ``label`` rather than
    subsuming it: ``label`` is a plain column so ``WHERE label = ...`` still
    works, while metadata carries whatever structure the analysis needs.
    """

    model: str
    label: str
    metadata: Metadata
    created_utc: str
    logits: SequenceLogits


def logits_key(sequence: str, start: int, stop: int) -> str:
    """Content-addressed key for one scored request, the second half of the PK.

    Sequences run to thousands of residues, so the row is keyed by a fixed-size
    digest rather than the sequence itself; the sequence is stored in its own
    column and never compared. Hashing keeps ``model`` a free-standing key
    column, so one request has one key across every model and a single
    ``WHERE sequence_key = ?`` answers which models have scored it.

    The digest is over the whole submitted sequence plus the region, which is
    what makes the same residues read out of two different sequences two
    different results: the context differs, so the logits differ.

    blake2b is fixed across processes, unlike the salted builtin ``hash()``, so
    a recompute lands on the same row instead of inserting a duplicate.
    """
    return hashlib.blake2b(
        f"{sequence}:{start}-{stop}".encode(), digest_size=16
    ).hexdigest()


class LogitsStorage(Protocol):
    """Persistence seam for computed masked-logits.

    Implementations bind to one resource. :class:`SqliteLogitsStorage` and
    :class:`PostgresLogitsStorage` are the two; both satisfy this contract
    identically, so callers never branch on which one they hold.
    """

    def has(self, *, model: str, sequence: str, start: int, stop: int) -> bool:
        """Whether an entry exists, without transferring the logits blob."""
        ...

    def load(self, *, model: str, sequence: str, start: int, stop: int) -> StoredLogits:
        """Returns the stored entry; raises :class:`KeyError` when absent."""
        ...

    def save(
        self,
        result: SequenceLogits,
        *,
        model: str,
        label: str,
        metadata: Metadata,
    ) -> None:
        """Persists ``result`` under ``(model, sequence, region)``."""
        ...

    def entries(self, *, model: str) -> Iterator[StoredLogits]:
        """Yields every stored entry for ``model``, ordered by sequence key."""
        ...

    def models(self) -> list[tuple[str, int]]:
        """Every model the store holds, with its entry count, by model name."""
        ...


type Row = tuple[object, ...]


class _Cursor(Protocol):
    """The slice of DB-API 2.0 cursor both drivers expose identically."""

    @property
    def description(self) -> object:
        """``None`` for statements that returned no result set."""
        ...

    def execute(self, statement: LiteralString, parameters: Row, /) -> object: ...

    def fetchall(self) -> Sequence[Row]: ...

    def close(self) -> None: ...


class _Connection(Protocol):
    """The slice of DB-API 2.0 connection both drivers expose identically."""

    def cursor(self) -> _Cursor: ...

    def commit(self) -> None: ...


class _SqlLogitsStorage:
    """Shared SQL implementation over an open DB-API connection.

    Subclasses supply the connection, the placeholder style, and the blob
    column type; everything else — schema, queries, row codec — is identical,
    because both engines speak the same ``ON CONFLICT ... DO UPDATE`` upsert.
    Statements are written with ``?`` and rewritten for the engine's paramstyle.

    Every statement is typed :class:`LiteralString` end to end — the table
    name, the blob type and the placeholder are all literals — so psycopg's
    injection guard proves no caller value can reach the SQL text. Values
    always travel as bound parameters.
    """

    _placeholder: LiteralString
    _blob_type: LiteralString

    def __init__(self, connection: _Connection) -> None:
        self._connection = connection
        self._create_schema()

    def _sql(self, statement: LiteralString) -> LiteralString:
        """Rewrites ``?`` placeholders into the engine's paramstyle."""
        return statement.replace("?", self._placeholder)

    def _execute(self, statement: LiteralString, parameters: Row = ()) -> list[Row]:
        """Runs one statement, commits, and returns every row it produced.

        Reads are materialized rather than streamed so the cursor can be closed
        here: entry iteration is a generator, and a half-consumed one must not
        pin a cursor open on the connection.
        """
        cursor = self._connection.cursor()
        try:
            cursor.execute(self._sql(statement), parameters)
            rows = list(cursor.fetchall()) if cursor.description else []
        finally:
            cursor.close()
        self._connection.commit()
        return rows

    def _create_schema(self) -> None:
        """Creates the table if absent, so a fresh database needs no migration step."""
        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                model TEXT NOT NULL,
                sequence_key TEXT NOT NULL,
                sequence TEXT NOT NULL,
                label TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_utc TEXT NOT NULL,
                region_start INTEGER NOT NULL,
                region_stop INTEGER NOT NULL,
                n_positions INTEGER NOT NULL,
                n_tokens INTEGER NOT NULL,
                vocab TEXT NOT NULL,
                logits {self._blob_type} NOT NULL,
                PRIMARY KEY (model, sequence_key)
            )
            """
        )

    def has(self, *, model: str, sequence: str, start: int, stop: int) -> bool:
        """Whether the row exists, selecting a constant so the blob stays in the database."""
        rows = self._execute(
            f"SELECT 1 FROM {TABLE} WHERE model = ? AND sequence_key = ?",
            (model, logits_key(sequence, start, stop)),
        )
        return bool(rows)

    def load(self, *, model: str, sequence: str, start: int, stop: int) -> StoredLogits:
        """Rehydrates the stored entry, raising :class:`KeyError` when absent."""
        rows = self._execute(
            f"{_SELECT_COLUMNS} FROM {TABLE} WHERE model = ? AND sequence_key = ?",
            (model, logits_key(sequence, start, stop)),
        )
        if not rows:
            raise KeyError(f"No stored logits for {model!r} and this sequence")
        return _row_to_stored(rows[0])

    def save(
        self,
        result: SequenceLogits,
        *,
        model: str,
        label: str,
        metadata: Metadata,
    ) -> None:
        """Upserts the entry in one statement, overwriting any prior one.

        A single statement is the whole atomicity story: a recompute replaces
        the row or does nothing, and there is no state in which a later run can
        see a half-written entry and skip it forever.
        """
        logits = np.ascontiguousarray(result.logits, dtype=np.float32)
        self._execute(
            f"""
            INSERT INTO {TABLE} (
                model, sequence_key, sequence, label, metadata, created_utc,
                region_start, region_stop, n_positions, n_tokens, vocab, logits
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (model, sequence_key) DO UPDATE SET
                sequence = EXCLUDED.sequence,
                label = EXCLUDED.label,
                metadata = EXCLUDED.metadata,
                created_utc = EXCLUDED.created_utc,
                region_start = EXCLUDED.region_start,
                region_stop = EXCLUDED.region_stop,
                n_positions = EXCLUDED.n_positions,
                n_tokens = EXCLUDED.n_tokens,
                vocab = EXCLUDED.vocab,
                logits = EXCLUDED.logits
            """,
            (
                model,
                logits_key(result.sequence, result.start, result.stop),
                result.sequence,
                label,
                json.dumps(dict(metadata)),
                datetime.now(UTC).isoformat(),
                result.start,
                result.stop,
                logits.shape[0],
                logits.shape[1],
                json.dumps(dict(result.vocab)),
                logits.tobytes(),
            ),
        )

    def entries(self, *, model: str) -> Iterator[StoredLogits]:
        """Yields every entry for ``model``; a model never written to yields nothing."""
        rows = self._execute(
            f"{_SELECT_COLUMNS} FROM {TABLE} WHERE model = ? ORDER BY sequence_key",
            (model,),
        )
        for row in rows:
            yield _row_to_stored(row)

    def models(self) -> list[tuple[str, int]]:
        """Every model the store holds, with its entry count, by model name.

        The store is keyed by ``(model, sequence)`` and nothing else records
        which models have been computed, so this is how a caller tells "no
        entries for this model" apart from "an empty store".
        """
        rows = self._execute(
            f"SELECT model, COUNT(*) FROM {TABLE} GROUP BY model ORDER BY model"
        )
        # The driver types every column as ``object``; both are known here.
        return [(str(row[0]), int(row[1])) for row in rows]  # type: ignore[arg-type]


class SqliteLogitsStorage(_SqlLogitsStorage):
    """SQLite storage bound to one database file.

    The default resource for every run: a single file, no server, and the
    logits blobs stay out of the working set until a report asks for them.
    """

    _placeholder = "?"
    _blob_type = "BLOB"

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        # WAL lets a report read the store while a compute run writes to it.
        connection.execute("PRAGMA journal_mode=WAL")
        super().__init__(connection)


class PostgresLogitsStorage(_SqlLogitsStorage):
    """PostgreSQL storage bound to one database.

    For a store shared by several machines — a Modal run and a local report
    against the same rows. ``psycopg`` is imported here rather than at module
    scope so a SQLite-only run never pays for it.
    """

    _placeholder = "%s"
    _blob_type = "BYTEA"

    def __init__(
        self, *, host: str, port: int, dbname: str, user: str, password: str
    ) -> None:
        import psycopg

        super().__init__(
            psycopg.connect(
                host=host, port=port, dbname=dbname, user=user, password=password
            )
        )


_SELECT_COLUMNS: LiteralString = """SELECT
    model, sequence, label, metadata, created_utc,
    region_start, region_stop, n_positions, n_tokens, vocab, logits"""


def _row_to_stored(row: Row) -> StoredLogits:
    """Decodes one selected row into a :class:`StoredLogits`.

    The blob is raw C-order float32 whose shape lives in its own columns, so
    the decode is a reshape rather than a deserialization. It is copied because
    ``frombuffer`` yields a read-only view over memory the driver owns.
    """
    (
        model,
        sequence,
        label,
        metadata,
        created_utc,
        region_start,
        region_stop,
        n_positions,
        n_tokens,
        vocab,
        blob,
    ) = row
    logits: npt.NDArray[np.float32] = (
        np.frombuffer(blob, dtype=np.float32)  # type: ignore[arg-type]
        .reshape(int(n_positions), int(n_tokens))  # type: ignore[arg-type]
        .copy()
    )
    return StoredLogits(
        model=str(model),
        label=str(label),
        metadata=json.loads(str(metadata)),
        created_utc=str(created_utc),
        logits=SequenceLogits(
            sequence=str(sequence),
            start=int(region_start),  # type: ignore[arg-type]
            stop=int(region_stop),  # type: ignore[arg-type]
            logits=logits,
            vocab=json.loads(str(vocab)),
        ),
    )


@dataclass(frozen=True)
class StorageSettings:
    """Fully-resolved storage configuration, built from the CLI by both scripts.

    Carries every backend's parameters at once — only those of ``storage`` are
    populated — so the two entrypoints pass one value around instead of a
    widening argument list.
    """

    storage: str
    sqlite_path: Path
    postgres_host: str
    postgres_port: int
    postgres_dbname: str
    postgres_user: str
    postgres_password: str

    def describe(self) -> str:
        """Human-readable identity of the resource, safe to print and to log.

        The password is deliberately absent: this string goes to stdout and
        into the longitudinal perf CSV.
        """
        if self.storage == "sqlite":
            return f"sqlite:{self.sqlite_path}"
        return (
            f"postgres:{self.postgres_user}@{self.postgres_host}:"
            f"{self.postgres_port}/{self.postgres_dbname}"
        )

    def flags(self) -> str:
        """The flags a sibling command needs to reach this same resource."""
        if self.storage == "sqlite":
            return f"--storage sqlite --sqlite-path {self.sqlite_path}"
        return (
            f"--storage postgres --postgres-host {self.postgres_host} "
            f"--postgres-port {self.postgres_port} "
            f"--postgres-dbname {self.postgres_dbname}"
        )


def add_storage_arguments(
    parser: argparse.ArgumentParser, *, sqlite_default: Path
) -> None:
    """Declares ``--storage`` and every backend's flags on a script's parser.

    Shared by every entrypoint so all stages accept the same storage vocabulary
    and a command line copied between them keeps working. Defaults of ``None``
    let :func:`resolve_params` distinguish "not given" from "given"; the one
    exception is ``sqlite_default``, which a script passes because the database
    it reads and writes is its topic's.

    ``--storage`` selects sqlite (the default, at ``sqlite_default``) or
    postgres. SQLite takes only ``--sqlite-path``; postgres takes its
    connection parameters, each with a ``POSTGRES_*`` env fallback listed in
    ``.env.example``.
    """
    parser.add_argument(
        "--storage",
        choices=STORAGES,
        default="sqlite",
        help="Storage backend holding computed logits (default: sqlite)",
    )
    for specs in storage_params(sqlite_default).values():
        for spec in specs:
            parser.add_argument(
                spec.flag,
                dest=spec.dest,
                default=None,
                type=spec.type,
                choices=spec.choices,
                help=spec.help,
            )


def storage_settings(
    args: argparse.Namespace, *, sqlite_default: Path
) -> StorageSettings:
    """Resolves the parsed namespace into a :class:`StorageSettings`.

    ``sqlite_default`` is the same one the script gave
    :func:`add_storage_arguments`, so what ``--help`` printed is what an
    omitted flag resolves to. Delegates to :func:`resolve_params`, which
    applies the env fallbacks and rejects a flag belonging to the storage that
    was *not* selected — so a stale ``--sqlite-path`` on a postgres run is an
    error rather than being silently ignored. Raises :class:`ValueError`, which
    every script surfaces as an argparse error.
    """
    cli_values: dict[str, ParamValue | None] = {
        "sqlite_path": args.sqlite_path,
        "postgres_host": args.postgres_host,
        "postgres_port": args.postgres_port,
        "postgres_dbname": args.postgres_dbname,
        "postgres_user": args.postgres_user,
        "postgres_password": args.postgres_password,
    }
    resolved = resolve_params(
        storage_params(sqlite_default)[args.storage],
        cli_values,
        label=f"storage {args.storage!r}",
    )
    return StorageSettings(
        storage=args.storage,
        sqlite_path=cast(Path, resolved.get("sqlite_path", sqlite_default)),
        postgres_host=cast(str, resolved.get("postgres_host", "")),
        postgres_port=cast(int, resolved.get("postgres_port", 0)),
        postgres_dbname=cast(str, resolved.get("postgres_dbname", "")),
        postgres_user=cast(str, resolved.get("postgres_user", "")),
        postgres_password=cast(str, resolved.get("postgres_password", "")),
    )


def open_storage(settings: StorageSettings) -> LogitsStorage:
    """Opens the storage selected by ``settings``, creating its schema if needed."""
    match settings.storage:
        case "sqlite":
            return SqliteLogitsStorage(settings.sqlite_path)
        case "postgres":
            return PostgresLogitsStorage(
                host=settings.postgres_host,
                port=settings.postgres_port,
                dbname=settings.postgres_dbname,
                user=settings.postgres_user,
                password=settings.postgres_password,
            )
        case _:
            raise ValueError(
                f"Unknown storage {settings.storage!r}; expected one of {STORAGES}"
            )


def select_models(storage: LogitsStorage, model: str | None) -> list[str]:
    """Which models a consuming stage runs over, or a :class:`ValueError` saying why none.

    ``None`` means every model the store holds: consuming a store costs no
    model time, so the useful default is "everything you have". Asking for a
    model with nothing stored is still an error, since it is almost always a
    compute run that used a different one, and the message names what the store
    does hold because the useful next command is the same one with a different
    ``--model`` or with none at all.

    Shared by every report topic, so they agree on the default and fail
    identically.
    """
    stored = storage.models()
    if not stored:
        raise ValueError(
            "Storage holds no entries at all, so there is nothing to report"
        )

    names = [name for name, _ in stored]
    if model is None:
        return names
    if model not in names:
        held = ", ".join(f"{name} ({count})" for name, count in stored)
        raise ValueError(
            f"Storage holds no entries for model {model!r}; it holds: {held}. "
            f"Re-run with a --model it holds, or without --model for all of them."
        )
    return [model]
