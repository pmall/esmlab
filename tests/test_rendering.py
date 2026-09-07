"""Covers what the two report topics must agree on: the index row's shared columns."""

from esmlab import peptides_render, sequences_render
from esmlab.peptides_scoring import analyze as analyze_peptide
from esmlab.rendering import Payload
from esmlab.sequences_scoring import analyze as analyze_sequence
from esmlab.storage import StoredLogits
from tests.fixtures import make_result, one_hot_rows

SEQUENCE = "ACDEFGHIKL"

# What both listing pages draw from a row: the link, the residue span, and when
# it was computed. The templates share that code, so the payloads must agree.
SHARED_ROW_KEYS = {
    "key",
    "label",
    "length",
    "start",
    "stop",
    "full_length",
    "created_utc",
}


def _entry(label: str = "tiny") -> StoredLogits:
    """One stored entry both topics can be asked to render."""
    return StoredLogits(
        backend="stub",
        model="esmc-600m",
        label=label,
        metadata={},
        created_utc="2026-01-01T00:00:00+00:00",
        logits=make_result(SEQUENCE, one_hot_rows(SEQUENCE, {})),
    )


def _peptides_row() -> Payload:
    entry = _entry()
    payload = peptides_render.entry_payload(
        entry, analyze_peptide(entry.logits), threshold=0.8, top_k=5
    )
    return peptides_render.index_payload([payload])["entries"][0]


def _sequences_row() -> Payload:
    entry = _entry()
    payload = sequences_render.entry_payload(
        entry, analyze_sequence(entry.logits, window=3)
    )
    return sequences_render.index_payload([payload])["entries"][0]


def test_both_index_rows_carry_the_columns_the_shared_skeleton_draws() -> None:
    """Neither topic may drop a column its listing template reads from every row."""
    assert SHARED_ROW_KEYS <= _peptides_row().keys()
    assert SHARED_ROW_KEYS <= _sequences_row().keys()


def test_the_two_rows_agree_on_those_columns_for_one_entry() -> None:
    """The same stored entry lists identically under either topic."""
    peptides = _peptides_row()
    sequences = _sequences_row()

    assert {key: peptides[key] for key in SHARED_ROW_KEYS} == {
        key: sequences[key] for key in SHARED_ROW_KEYS
    }
