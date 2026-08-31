from pathlib import Path

import pytest

from esmlab.seqio import parse_sequences, validate_sequence


def test_validate_sequence_normalizes_and_rejects() -> None:
    """Sequences are trimmed/uppercased and non-canonical or empty input raises."""
    assert validate_sequence(" acde ") == "ACDE"
    with pytest.raises(ValueError, match="non-canonical"):
        validate_sequence("ACXZ")
    with pytest.raises(ValueError, match="Empty"):
        validate_sequence("   ")


def test_parse_sequences_labels_positional_inputs_and_sanitizes_headers(
    tmp_path: Path,
) -> None:
    """Positional inputs get seq_NN labels, FASTA headers are sanitized, empty raises."""
    fasta = tmp_path / "input.fa"
    fasta.write_text(">my protein|1\nACDE\nFGHIK\n\n>dup-name\nMMMMM\n")

    named = parse_sequences(["acdefghikl"], [fasta])
    assert [record.name for record in named] == ["seq_01", "my_protein_1", "dup-name"]
    assert named[0].sequence == "ACDEFGHIKL"

    with pytest.raises(ValueError, match="No input sequences"):
        parse_sequences([], [])


def test_parse_sequences_accepts_duplicate_labels(tmp_path: Path) -> None:
    """Labels are display-only, so two records may share one without conflict."""
    fasta = tmp_path / "input.fa"
    fasta.write_text(">dup-name\nMMMMM\n")
    other = tmp_path / "other.fa"
    other.write_text(">dup-name\nLLLLL\n")

    named = parse_sequences([], [fasta, other])

    assert [record.name for record in named] == ["dup-name", "dup-name"]
    assert [record.sequence for record in named] == ["MMMMM", "LLLLL"]
