"""Covers the whole-sequence render stage: the payload's contract and the ranking index."""

import pytest

from esmlab.connectors.base import SequenceLogits
from esmlab.rendering import PAYLOAD_MARKER, Payload
from esmlab.sequences_render import (
    ENTROPY_LIMIT,
    entry_payload,
    index_payload,
    render_entry,
    render_index,
)
from esmlab.sequences_scoring import analyze
from esmlab.storage import StoredLogits
from tests.fixtures import make_result, one_hot_rows, uniform_rows

SEQUENCE = "ACDEFGHIKL"


def _payload_for(result: SequenceLogits, label: str = "tiny") -> Payload:
    """The payload for one handcrafted result, as the report stage builds it."""
    entry = StoredLogits(
        backend="stub",
        model="esmc-600m",
        label=label,
        metadata={},
        created_utc="2026-01-01T00:00:00+00:00",
        logits=result,
    )
    return entry_payload(entry, analyze(result, window=3))


def test_payload_carries_one_row_per_scored_residue_and_no_matrix() -> None:
    """The entropy page is one number per position: no substitution matrix."""
    payload = _payload_for(make_result(SEQUENCE, one_hot_rows(SEQUENCE, {})))

    assert payload["length"] == len(SEQUENCE)
    assert len(payload["entropy"]) == len(SEQUENCE)
    assert len(payload["smoothed"]) == len(SEQUENCE)
    assert payload["positions"] == list(range(1, len(SEQUENCE) + 1))
    assert "llr" not in payload


def test_highlights_run_from_the_most_certain_to_the_most_open_position() -> None:
    """The two lists are the extremes of the same track, in opposite order."""
    hot = {index: list("ACDEFGHIKLMNPQRST") for index in (3,)}
    payload = _payload_for(make_result(SEQUENCE, one_hot_rows(SEQUENCE, hot)))

    assert payload["most_variable"][0]["position"] == 4
    assert (
        payload["most_variable"][0]["value"] > payload["most_constrained"][0]["value"]
    )
    assert payload["most_constrained"][0]["value"] == pytest.approx(0.0)


def test_summary_scalars_are_the_ones_the_index_ranks_on() -> None:
    """A page's own numbers and its row in the listing come from one shape."""
    payload = _payload_for(make_result("AAAA", uniform_rows("AAAA", 20)))

    assert payload["summary"]["mean"] == pytest.approx(ENTROPY_LIMIT, abs=1e-4)
    assert payload["entropy_limit"] == pytest.approx(ENTROPY_LIMIT, abs=1e-4)
    row = index_payload([payload])["entries"][0]
    assert row["mean"] == payload["summary"]["mean"]


def test_index_ranks_the_most_constrained_sequence_first() -> None:
    """The listing is the comparison, so it orders by mean entropy, not by time."""
    constrained = _payload_for(make_result("AAAA", one_hot_rows("AAAA", {})), "sharp")
    free = _payload_for(make_result("AAAA", uniform_rows("AAAA", 20)), "flat")

    payload = index_payload([free, constrained])

    assert [row["label"] for row in payload["entries"]] == ["sharp", "flat"]
    assert payload["model"] == "esmc-600m"


def test_both_pages_are_filled_from_their_own_templates() -> None:
    """The entropy topic has its own two templates, and neither is left unfilled."""
    payload = _payload_for(make_result(SEQUENCE, one_hot_rows(SEQUENCE, {})))

    page = render_entry(payload)
    assert PAYLOAD_MARKER not in page
    assert "sequence report" in page
    assert PAYLOAD_MARKER not in render_index(index_payload([payload]))
