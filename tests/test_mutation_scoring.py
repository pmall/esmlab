import math

import numpy as np
import pytest

from esmlab.amino_acids import VALID_AMINO_ACIDS
from esmlab.mutation_scoring import (
    deleterious_fraction_per_position,
    entropy_per_position,
    llr_matrix,
    rank_substitutions,
    tolerant_positions,
    uniform_entropy_bits,
)
from tests.fixtures import make_result, one_hot_rows, uniform_rows

VOCAB = len(VALID_AMINO_ACIDS)


def test_uniform_entropy_matches_log2_of_alphabet() -> None:
    for alphabet_size in (2, 4, 8, 16, 20):
        result = make_result("AAA", uniform_rows("AAA", alphabet_size))
        entropies = entropy_per_position(result)
        assert entropies == pytest.approx([math.log2(alphabet_size)] * 3)


def test_sharp_distribution_has_zero_entropy() -> None:
    result = make_result("AC", one_hot_rows("AC", {}))
    assert entropy_per_position(result) == pytest.approx([0.0, 0.0])


def test_llr_wildtype_column_is_zero_and_preferred_alternative_is_positive() -> None:
    # Position 0: model prefers C and G over the wildtype A. Position 1: flat.
    rows = [
        [1.0 if aa in ("C", "G") else 0.0 for aa in VALID_AMINO_ACIDS],
        [0.0] * VOCAB,
    ]
    llr = llr_matrix(make_result("AA", rows))

    assert llr[0, VALID_AMINO_ACIDS.index("A")] == pytest.approx(0.0)
    assert llr[0, VALID_AMINO_ACIDS.index("C")] > 0.0
    assert llr[1] == pytest.approx(np.zeros(VOCAB))


def test_deleterious_fraction_counts_negative_non_wildtype_entries() -> None:
    llr = np.zeros((2, VOCAB))
    for row in range(2):
        wt_column = (3, 7)[row]
        alternatives = [c for c in range(VOCAB) if c != wt_column]
        for column in alternatives[:5]:
            llr[row, column] = -1.0

    fractions = deleterious_fraction_per_position(llr)
    assert fractions == pytest.approx([5 / 19, 5 / 19])


def test_tolerant_positions_uses_strict_threshold_comparison() -> None:
    fractions = np.array([0.10, 0.80, 0.79])
    assert tolerant_positions(fractions, threshold=0.8).tolist() == [0, 2]


def test_rank_substitutions_orders_descending_and_skips_wildtype() -> None:
    llr = np.full((2, VOCAB), -0.5)
    llr[0, VALID_AMINO_ACIDS.index("W")] = 2.0
    llr[0, VALID_AMINO_ACIDS.index("V")] = 1.0

    ranked = rank_substitutions(llr, "AA", top_k=4)

    assert ranked[0] == (1, "A", "W", 2.0)
    assert ranked[1] == (1, "A", "V", 1.0)
    assert all(substitution[1] != substitution[2] for substitution in ranked)
    scores = [substitution[3] for substitution in ranked]
    assert scores == sorted(scores, reverse=True)


def test_uniform_entropy_bits_helper() -> None:
    assert uniform_entropy_bits(20) == math.log2(20)
