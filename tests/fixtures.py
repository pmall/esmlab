"""Builds small handcrafted SequenceLogits fixtures for pure-math tests."""

import numpy as np
import numpy.typing as npt

from esmlab.amino_acids import VALID_AMINO_ACIDS
from esmlab.connectors.base import SequenceLogits
from esmlab.connectors.stub import STUB_VOCAB

VOCAB_SIZE = len(VALID_AMINO_ACIDS)


def make_result(sequence: str, rows: npt.ArrayLike) -> SequenceLogits:
    """Builds a :class:`SequenceLogits` fixture from handcrafted logit rows.

    ``rows`` must have shape ``(len(sequence), VOCAB_SIZE)``; the vocab is the
    stub's (:data:`STUB_VOCAB`). Used by the pure-math tests in
    :mod:`tests.test_mutation_scoring` to drive the scoring functions without
    a model.
    """
    logits = np.asarray(rows, dtype=np.float32)
    assert logits.shape == (len(sequence), VOCAB_SIZE)
    return SequenceLogits(sequence=sequence, logits=logits, vocab=dict(STUB_VOCAB))


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
