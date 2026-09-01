from pathlib import Path

import pytest

from esmlab.seqio import parse_header, parse_sequences, validate_sequence


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


def test_header_without_metadata_yields_an_empty_object() -> None:
    """The JSON part is optional, so a plain header still parses."""
    assert parse_header("stat1", "fallback") == ("stat1", {})


def test_header_carries_arbitrary_json_metadata() -> None:
    """`label|{...}` is the whole mechanism for adding structure to a record."""
    label, metadata = parse_header(
        'stat1|{"source": "UniProt:P12345", "targets": ["P11111", "P22222"]}',
        "fallback",
    )

    assert label == "stat1"
    assert metadata == {
        "source": "UniProt:P12345",
        "targets": ["P11111", "P22222"],
    }


def test_pipe_separated_identifiers_stay_in_the_label() -> None:
    """Splitting on `|{` and not `|` keeps conventional ids whole."""
    assert parse_header("sp|P12345|STAT1_HUMAN", "fallback") == (
        "sp_P12345_STAT1_HUMAN",
        {},
    )

    label, metadata = parse_header('sp|P12345|STAT1_HUMAN|{"n": 1}', "fallback")
    assert (label, metadata) == ("sp_P12345_STAT1_HUMAN", {"n": 1})


def test_malformed_header_json_is_reported_with_its_location(tmp_path: Path) -> None:
    """A typo fails the run instead of silently storing no metadata."""
    fasta = tmp_path / "input.fa"
    fasta.write_text('>stat1|{"source": }\nACDE\n')

    with pytest.raises(ValueError, match=r"input.fa:1: invalid JSON metadata"):
        parse_sequences([], [fasta])


def test_fasta_metadata_reaches_the_parsed_record(tmp_path: Path) -> None:
    """Header JSON survives the full parse, ready for storage."""
    fasta = tmp_path / "input.fa"
    fasta.write_text('>stat1|{"targets": ["P11111"]}\nACDE\n>stat2\nMMMMM\n')

    records = parse_sequences([], [fasta])

    assert [record.name for record in records] == ["stat1", "stat2"]
    assert records[0].metadata == {"targets": ["P11111"]}
    assert records[1].metadata == {}
