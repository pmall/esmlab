"""Covers the render stage: the payload's contract and the template filling."""

import json

import pytest

from esmlab.amino_acids import AA_TO_ISOELECTRIC_POINT, VALID_AMINO_ACIDS
from esmlab.connectors.base import SequenceLogits
from esmlab.connectors.stub import STUB_VOCAB
from esmlab.peptides_render import (
    entry_payload,
    index_payload,
    render_entry,
    render_index,
)
from esmlab.peptides_scoring import analyze
from esmlab.rendering import PAYLOAD_MARKER, Payload
from esmlab.storage import StoredLogits
from tests.fixtures import make_result, one_hot_rows

SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"


def _payload_for(result: SequenceLogits, label: str = "tiny") -> Payload:
    """The payload for one handcrafted result, as the report stage builds it."""
    entry = StoredLogits(
        model="esmc-600m",
        label=label,
        metadata={},
        created_utc="2026-01-01T00:00:00Z",
        logits=result,
    )
    return entry_payload(entry, analyze(result), threshold=0.8, top_k=5)


def _payload(label: str = "tiny") -> Payload:
    """A payload for a sequence whose position 0 tolerates one substitution."""
    return _payload_for(
        make_result(SEQUENCE, one_hot_rows(SEQUENCE, {0: ["A", "W"]})), label
    )


def _graded_rows() -> list[list[float]]:
    """Rows spread over ~2 nats, so every LLR is small and none dominates."""
    return [
        [0.1 * column for column in range(len(VALID_AMINO_ACIDS))] for _ in SEQUENCE
    ]


def test_matrix_columns_follow_the_row_order_the_page_draws() -> None:
    """LLR columns are re-ordered to the payload's own amino-acid order.

    The page indexes ``llr[position][i]`` with the same ``i`` it uses for
    ``amino_acids``, so the reordering has to happen here rather than in the
    template.
    """
    payload = _payload()
    amino_acids = payload["amino_acids"]

    assert amino_acids == sorted(
        VALID_AMINO_ACIDS, key=lambda aa: AA_TO_ISOELECTRIC_POINT[aa], reverse=True
    )
    # Position 0 is wildtype A with W equally likely: both sit at LLR 0.
    row = payload["llr"][0]
    assert row[amino_acids.index("A")] == 0.0
    assert row[amino_acids.index("W")] == 0.0
    assert row[amino_acids.index("D")] < 0.0


def test_payload_is_json_safe_inside_a_script_element() -> None:
    """A label that closes the script element cannot break the JSON island.

    Labels are sanitized FASTA headers, so they are arbitrary text; the page
    is unreadable if one of them ends the payload's ``<script>`` early.
    """
    page = render_entry(_payload(label="</script><script>alert(1)"))

    assert "</script><script>alert(1)" not in page
    assert PAYLOAD_MARKER not in page
    island = page.split('<script type="application/json" id="payload">')[1]
    island = island.split("</script>")[0]
    assert json.loads(island.replace("<\\/", "</"))["label"] == (
        "</script><script>alert(1)"
    )


def test_index_lists_newest_first_under_its_model() -> None:
    """The listing is ordered by compute time and names the model it covers."""
    older = _payload(label="older")
    older["created_utc"] = "2025-01-01T00:00:00Z"

    payload = index_payload([older, _payload(label="newer")])

    assert payload["model"] == "esmc-600m"
    assert [row["label"] for row in payload["entries"]] == ["newer", "older"]
    assert PAYLOAD_MARKER not in render_index(payload)


def test_color_scale_ignores_a_single_extreme_substitution() -> None:
    """One catastrophic LLR must not flatten the rest of the matrix to neutral.

    The page divides by the color scale, so scaling to the maximum would put
    every ordinary cell within a few percent of the midpoint and the matrix
    would read as blank.
    """
    rows = _graded_rows()
    ordinary = _payload_for(make_result(SEQUENCE, rows))
    rows[3][STUB_VOCAB["D"]] = -60.0
    outlier = _payload_for(make_result(SEQUENCE, rows))

    assert outlier["llr_max_abs"] > 10 * outlier["llr_color_scale"]
    # The outlier moved the largest value by 30x and the color scale barely at all.
    assert outlier["llr_max_abs"] > 10 * ordinary["llr_max_abs"]
    assert outlier["llr_color_scale"] == pytest.approx(
        ordinary["llr_color_scale"], rel=0.25
    )


def test_color_scale_stays_positive_on_a_flat_matrix() -> None:
    """An all-zero matrix still yields a divisor the page can use."""
    flat = make_result(SEQUENCE, [[0.0] * len(VALID_AMINO_ACIDS)] * len(SEQUENCE))

    assert _payload_for(flat, "flat")["llr_color_scale"] > 0
