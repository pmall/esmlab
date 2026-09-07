"""Covers the per-position amino-acid distributions every logits topic reads."""

import math

import pytest

from esmlab.distributions import (
    aa_log_probs,
    entropy_per_position,
    uniform_entropy_bits,
)
from tests.fixtures import make_result, one_hot_rows, uniform_rows


def test_uniform_entropy_matches_log2_of_alphabet() -> None:
    """Uniform-over-k rows must yield entropy log2(k) at every position."""
    for alphabet_size in (2, 4, 8, 16, 20):
        result = make_result("AAA", uniform_rows("AAA", alphabet_size))
        entropies = entropy_per_position(result)
        assert entropies == pytest.approx([math.log2(alphabet_size)] * 3)


def test_sharp_distribution_has_zero_entropy() -> None:
    """A one-hot distribution (all mass on the wildtype) has zero entropy."""
    result = make_result("AC", one_hot_rows("AC", {}))
    assert entropy_per_position(result) == pytest.approx([0.0, 0.0])


def test_log_probs_are_normalized_over_the_canonical_residues() -> None:
    """Each row is a distribution over the 20 residues, special tokens excluded."""
    result = make_result("AC", one_hot_rows("AC", {0: ["A", "C", "D"]}))
    probabilities = pytest.approx(1.0)
    for row in aa_log_probs(result):
        assert float(sum(math.exp(value) for value in row)) == probabilities


def test_uniform_entropy_bits_helper() -> None:
    """The reference helper returns log2 of the alphabet size."""
    assert uniform_entropy_bits(20) == math.log2(20)
