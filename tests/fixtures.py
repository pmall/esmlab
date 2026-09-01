"""Shared test fixtures: handcrafted SequenceLogits and a SQLite storage config."""

from pathlib import Path

import numpy as np
import numpy.typing as npt

from esmlab.amino_acids import VALID_AMINO_ACIDS
from esmlab.connectors.base import SequenceLogits
from esmlab.connectors.stub import STUB_VOCAB
from esmlab.seqio import NamedSequence
from esmlab.storage import StorageSettings

VOCAB_SIZE = len(VALID_AMINO_ACIDS)


def make_result(
    sequence: str,
    rows: npt.ArrayLike,
    start: int = 1,
    stop: int | None = None,
) -> SequenceLogits:
    """Builds a :class:`SequenceLogits` fixture from handcrafted logit rows.

    ``rows`` must have one row per scored residue, the region defaulting to the
    whole sequence; the vocab is the stub's (:data:`STUB_VOCAB`). Used by the
    pure-math tests in :mod:`tests.test_mutation_scoring` to drive the scoring
    functions without a model.
    """
    stop = len(sequence) if stop is None else stop
    logits = np.asarray(rows, dtype=np.float32)
    assert logits.shape == (stop - start + 1, VOCAB_SIZE)
    return SequenceLogits(
        sequence=sequence,
        start=start,
        stop=stop,
        logits=logits,
        vocab=dict(STUB_VOCAB),
    )


def named(
    name: str, sequence: str, start: int = 1, stop: int | None = None
) -> NamedSequence:
    """A :class:`NamedSequence` covering the whole sequence unless told otherwise.

    Records carry their region as two required fields; most tests do not care
    which residues, so this fills in "all of them".
    """
    return NamedSequence(name, sequence, start, len(sequence) if stop is None else stop)


def one_hot_rows(sequence: str, hot: dict[int, list[str]]) -> list[list[float]]:
    """Near-one-hot probability rows: logit +1 on listed amino acids, -100 elsewhere.

    Positions not in ``hot`` put all mass implicitly on their wildtype residue.
    The returned rows are meant to be passed to :func:`make_result`.
    """
    rows: list[list[float]] = []
    for position, wildtype in enumerate(sequence):
        row = [-100.0] * VOCAB_SIZE
        for aa in hot.get(position, [wildtype]):
            row[STUB_VOCAB[aa]] = 1.0
        rows.append(row)
    return rows


def uniform_rows(sequence: str, alphabet_size: int) -> list[list[float]]:
    """Rows equal over the first ``alphabet_size`` amino acids (alphabet order).

    Produces a uniform distribution of the requested alphabet size so entropy
    tests can compare against ``log2(alphabet_size)``. Passed to
    :func:`make_result`.
    """
    rows: list[list[float]] = []
    for _ in sequence:
        row = [-100.0] * VOCAB_SIZE
        for column in range(alphabet_size):
            row[column] = 0.0
        rows.append(row)
    return rows


def sqlite_settings(path: Path) -> StorageSettings:
    """A :class:`StorageSettings` selecting a SQLite file, with the rest blank.

    Tests build storage through the same settings object the CLI produces, so
    they exercise the real :func:`open_storage` path; only the SQLite fields
    matter when ``storage`` is ``"sqlite"``.
    """
    return StorageSettings(
        storage="sqlite",
        sqlite_path=path,
        postgres_host="",
        postgres_port=0,
        postgres_dbname="",
        postgres_user="",
        postgres_password="",
    )


def whole(sequence: str) -> dict[str, int]:
    """``start``/``stop`` keyword arguments covering an entire sequence.

    Storage is addressed by sequence *and* region, and most tests scored the
    whole thing; spelling that out at every call site would bury what they are
    actually asserting.
    """
    return {"start": 1, "stop": len(sequence)}
