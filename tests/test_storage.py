"""Covers the SQLite logits storage: keys, round-trip, upsert, isolation.

Everything here exercises :class:`SqliteLogitsStorage`, which shares its
schema, queries and row codec with :class:`PostgresLogitsStorage`; only the
paramstyle, blob type and connection differ, so these tests cover the SQL
behavior of both.
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from esmlab.connectors.stub import StubConnector
from esmlab.storage import (
    TABLE,
    SqliteLogitsStorage,
    logits_key,
)

SEQUENCE = "MKTAYIAKQRQISFVK"
OTHER_SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQ"
# Nested on purpose: metadata is an arbitrary JSON object, not a string map.
METADATA = {
    "source": "UniProt:P12345",
    "targets": ["P11111", "P22222"],
    "assay": {"replicates": 3, "validated": True},
}


def _result(sequence: str = SEQUENCE):
    """Deterministic logits for a sequence, without touching a real model."""
    return StubConnector("esmc-600m").masked_sequence_logits(sequence)


def _storage(tmp_path: Path) -> SqliteLogitsStorage:
    """A store bound to a fresh database file under the test's temp directory."""
    return SqliteLogitsStorage(tmp_path / "logits.sqlite3")


def test_missing_entry_is_absent_and_load_raises(tmp_path: Path) -> None:
    """An unwritten entry reports absent and refuses to load."""
    storage = _storage(tmp_path)

    assert storage.has(model="esmc-600m", sequence=SEQUENCE) is False
    with pytest.raises(KeyError):
        storage.load(model="esmc-600m", sequence=SEQUENCE)


def test_save_then_load_round_trips_result_and_provenance(tmp_path: Path) -> None:
    """A saved entry comes back with identical logits plus its display label."""
    storage = _storage(tmp_path)
    result = _result()

    storage.save(result, model="esmc-600m", label="stat1", metadata=METADATA)

    assert storage.has(model="esmc-600m", sequence=SEQUENCE) is True
    stored = storage.load(model="esmc-600m", sequence=SEQUENCE)
    assert stored.model == "esmc-600m"
    assert stored.label == "stat1"
    assert stored.metadata == METADATA
    assert stored.created_utc
    assert stored.logits.sequence == SEQUENCE
    assert stored.logits.vocab == dict(result.vocab)
    np.testing.assert_array_equal(stored.logits.logits, result.logits)


def test_loaded_logits_are_writable(tmp_path: Path) -> None:
    """The decoded array owns its memory, so scoring may work in place.

    ``np.frombuffer`` yields a read-only view over driver-owned memory; the
    decode copies precisely so a caller never hits that.
    """
    storage = _storage(tmp_path)
    storage.save(_result(), model="esmc-600m", label="a", metadata={})

    logits = storage.load(model="esmc-600m", sequence=SEQUENCE).logits.logits
    logits[0, 0] = 1.0

    assert logits[0, 0] == 1.0


def test_key_depends_on_sequence_alone(tmp_path: Path) -> None:
    """The digest is the sequence's; ``model`` is a free-standing key column."""
    assert logits_key(SEQUENCE) == logits_key(SEQUENCE)
    assert logits_key(SEQUENCE) != logits_key(OTHER_SEQUENCE)

    path = tmp_path / "logits.sqlite3"
    storage = SqliteLogitsStorage(path)
    storage.save(_result(), model="esmc-600m", label="a", metadata={})
    storage.save(_result(), model="esmc-300m", label="a", metadata={})

    # One sequence has one key everywhere, so a single row scan answers
    # which models have scored it.
    with sqlite3.connect(path) as connection:
        models = connection.execute(
            f"SELECT model FROM {TABLE} WHERE sequence_key = ? ORDER BY model",
            (logits_key(SEQUENCE),),
        ).fetchall()
    assert [row[0] for row in models] == ["esmc-300m", "esmc-600m"]


def test_entry_for_a_different_sequence_is_separate(tmp_path: Path) -> None:
    """Distinct sequences never share a row under the same model."""
    storage = _storage(tmp_path)
    storage.save(_result(), model="esmc-600m", label="a", metadata={})

    assert storage.has(model="esmc-600m", sequence=OTHER_SEQUENCE) is False


def test_saving_twice_upserts_in_place(tmp_path: Path) -> None:
    """A recompute lands on the same row rather than inserting a duplicate."""
    path = tmp_path / "logits.sqlite3"
    storage = SqliteLogitsStorage(path)
    storage.save(_result(), model="esmc-600m", label="first", metadata={})
    storage.save(_result(), model="esmc-600m", label="second", metadata={})

    with sqlite3.connect(path) as connection:
        count = connection.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    assert count == 1
    assert storage.load(model="esmc-600m", sequence=SEQUENCE).label == "second"


def test_row_holds_provenance_and_the_decoded_shape(tmp_path: Path) -> None:
    """The row describes the entry; the blob's shape lives in its own columns."""
    path = tmp_path / "logits.sqlite3"
    SqliteLogitsStorage(path).save(
        _result(), model="esmc-600m", label="stat1", metadata=METADATA
    )

    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(f"SELECT * FROM {TABLE}").fetchone()

    assert set(row.keys()) == {
        "model",
        "sequence_key",
        "sequence",
        "label",
        "metadata",
        "created_utc",
        "n_positions",
        "n_tokens",
        "vocab",
        "logits",
    }
    assert row["model"] == "esmc-600m"
    # The label stays a plain string, so WHERE label = 'stat1' still works.
    assert row["label"] == "stat1"
    assert json.loads(row["metadata"]) == METADATA
    assert row["sequence"] == SEQUENCE
    assert row["sequence_key"] == logits_key(SEQUENCE)
    assert (row["n_positions"], row["n_tokens"]) == _result().logits.shape
    assert json.loads(row["vocab"]) == dict(_result().vocab)


def test_a_second_store_on_the_same_file_sees_the_same_rows(tmp_path: Path) -> None:
    """The database is the only scope: two stores on one file are one store."""
    path = tmp_path / "logits.sqlite3"
    SqliteLogitsStorage(path).save(_result(), model="esmc-600m", label="a", metadata={})

    assert SqliteLogitsStorage(path).has(model="esmc-600m", sequence=SEQUENCE) is True


def test_entries_yields_only_the_requested_model(tmp_path: Path) -> None:
    """Enumeration is per-model."""
    storage = _storage(tmp_path)
    storage.save(_result(), model="esmc-600m", label="a", metadata={})
    storage.save(_result(OTHER_SEQUENCE), model="esmc-600m", label="b", metadata={})
    storage.save(_result(), model="esmc-300m", label="c", metadata={})

    labels = sorted(entry.label for entry in storage.entries(model="esmc-600m"))
    assert labels == ["a", "b"]
    assert [entry.label for entry in storage.entries(model="esmc-300m")] == ["c"]


def test_entries_on_unknown_model_is_empty(tmp_path: Path) -> None:
    """A model never written to yields nothing rather than raising."""
    assert list(_storage(tmp_path).entries(model="esmc-6b")) == []


def test_schema_is_created_on_a_fresh_database(tmp_path: Path) -> None:
    """Opening a store bootstraps its table, including the parent directory."""
    path = tmp_path / "nested" / "logits.sqlite3"
    SqliteLogitsStorage(path)

    assert path.is_file()
    with sqlite3.connect(path) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    assert TABLE in [row[0] for row in tables]


def test_metadata_round_trips_nested_structure(tmp_path: Path) -> None:
    """Lists and nested objects survive, so callers are not limited to flat strings."""
    storage = _storage(tmp_path)
    storage.save(_result(), model="esmc-600m", label="stat1", metadata=METADATA)

    stored = storage.load(model="esmc-600m", sequence=SEQUENCE)

    assert stored.metadata == METADATA
    assert stored.metadata["targets"] == ["P11111", "P22222"]


def test_metadata_survives_a_recompute(tmp_path: Path) -> None:
    """The upsert refreshes metadata rather than keeping the first run's copy."""
    storage = _storage(tmp_path)
    storage.save(_result(), model="esmc-600m", label="a", metadata={"targets": []})
    storage.save(_result(), model="esmc-600m", label="a", metadata=METADATA)

    assert storage.load(model="esmc-600m", sequence=SEQUENCE).metadata == METADATA


def test_metadata_is_queryable_without_decoding_in_python(tmp_path: Path) -> None:
    """Storing an object rather than an opaque blob keeps SQL able to filter on it."""
    path = tmp_path / "logits.sqlite3"
    storage = SqliteLogitsStorage(path)
    storage.save(_result(), model="esmc-600m", label="a", metadata=METADATA)
    storage.save(
        _result(OTHER_SEQUENCE),
        model="esmc-600m",
        label="b",
        metadata={"source": "UniProt:Q99999"},
    )

    with sqlite3.connect(path) as connection:
        labels = connection.execute(
            f"SELECT label FROM {TABLE} WHERE json_extract(metadata, '$.source') = ?",
            ("UniProt:P12345",),
        ).fetchall()
    assert [row[0] for row in labels] == ["a"]
