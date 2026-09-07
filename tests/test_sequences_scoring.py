"""Covers the whole-sequence topic's math: the smoothed track, the histogram, the summary."""

import math

import numpy as np
import pytest

from esmlab.sequences_scoring import (
    HISTOGRAM_BIN_BITS,
    analyze,
    histogram,
    rolling_mean,
)
from tests.fixtures import make_result, one_hot_rows, uniform_rows

LIMIT = math.log2(20)


def test_rolling_mean_shrinks_its_window_at_the_ends() -> None:
    """Every position gets a value, so the line plots against the same axis."""
    values = np.array([0.0, 3.0, 0.0, 3.0, 0.0])

    smoothed = rolling_mean(values, 3)

    assert smoothed == pytest.approx([1.5, 1.0, 2.0, 1.0, 1.5])


def test_a_window_of_one_leaves_the_track_alone() -> None:
    """The window is presentation only: at 1 the page draws the raw entropies."""
    values = np.array([0.0, 3.0, 1.0])

    assert rolling_mean(values, 1) == pytest.approx(values)


def test_histogram_spans_the_alphabet_ceiling_whatever_the_data_holds() -> None:
    """Bins are fixed by the alphabet, so two sequences share one axis."""
    bins = histogram(np.array([0.0, 0.0, LIMIT]), LIMIT)

    assert bins[0] == (0.0, 2)
    assert sum(count for _, count in bins) == 3
    assert bins[-1][0] == pytest.approx(LIMIT - LIMIT % HISTOGRAM_BIN_BITS)


def test_analyze_summarizes_a_uniform_sequence_at_the_ceiling() -> None:
    """Uniform-over-20 rows put every summary scalar on the ceiling."""
    result = make_result("AAAA", uniform_rows("AAAA", 20))

    analysis = analyze(result, window=3)

    assert analysis.mean == pytest.approx(LIMIT)
    assert analysis.median == pytest.approx(LIMIT)
    assert (analysis.minimum, analysis.maximum) == pytest.approx((LIMIT, LIMIT))
    assert analysis.smoothed == pytest.approx(analysis.entropies)


def test_analyze_carries_the_region_and_its_residues() -> None:
    """Rows describe the scored residues, whatever part of the sequence they are."""
    sequence = "ACDEFG"
    result = make_result(sequence, one_hot_rows("CDE", {}), start=2, stop=4)

    analysis = analyze(result, window=1)

    assert (analysis.residues, analysis.start, analysis.stop) == ("CDE", 2, 4)
    assert analysis.entropies == pytest.approx([0.0, 0.0, 0.0])
