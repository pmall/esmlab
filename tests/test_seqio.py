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
    fasta.write_text(">my protein|1|4\nACDE\nFGHIK\n\n>dup-name|1|4\nMMMMM\n")

    named = parse_sequences(["acdefghikl"], [fasta])
    assert [record.name for record in named] == ["seq_01", "my_protein", "dup-name"]
    assert named[0].sequence == "ACDEFGHIKL"

    with pytest.raises(ValueError, match="No input sequences"):
        parse_sequences([], [])


def test_parse_sequences_accepts_duplicate_labels(tmp_path: Path) -> None:
    """Labels are display-only, so two records may share one without conflict."""
    fasta = tmp_path / "input.fa"
    fasta.write_text(">dup-name|1|4\nMMMMM\n")
    other = tmp_path / "other.fa"
    other.write_text(">dup-name|1|4\nLLLLL\n")

    named = parse_sequences([], [fasta, other])

    assert [record.name for record in named] == ["dup-name", "dup-name"]
    assert [record.sequence for record in named] == ["MMMMM", "LLLLL"]


def test_header_carries_coordinates_and_optional_metadata() -> None:
    """Every header names a sub-sequence; the JSON object stays optional."""
    assert parse_header("nsp1|60|74", "fallback") == ("nsp1", 60, 74, {})
    assert parse_header('nsp1|60|74|{"strain": "x"}', "fallback") == (
        "nsp1",
        60,
        74,
        {"strain": "x"},
    )


def test_header_without_coordinates_is_rejected() -> None:
    """A record has to say which residues it is about, so a bare label fails."""
    for header in ("stat1", 'stat1|{"a": 1}', "stat1|60", "sp|P12345|STAT1_HUMAN"):
        with pytest.raises(ValueError, match="label|start|stop"):
            parse_header(header, "fallback")


def test_pipe_separated_identifiers_keep_their_pipes_in_the_label() -> None:
    """Only the last two fields are coordinates; the rest is the label."""
    assert parse_header("sp|P12345|STAT1_HUMAN|60|74", "fallback") == (
        "sp_P12345_STAT1_HUMAN",
        60,
        74,
        {},
    )


def test_malformed_header_json_is_reported_with_its_location(tmp_path: Path) -> None:
    """A typo fails the run instead of silently storing no metadata."""
    fasta = tmp_path / "input.fa"
    fasta.write_text('>stat1|1|4|{"source": }\nACDE\n')

    with pytest.raises(ValueError, match=r"input.fa:1: invalid JSON metadata"):
        parse_sequences([], [fasta])


def test_region_outside_the_sequence_is_reported_with_its_location(
    tmp_path: Path,
) -> None:
    """A coordinate past the sequence is a typo, not something to clamp."""
    fasta = tmp_path / "input.fa"
    fasta.write_text(">nsp1|3|99\nACDEF\n")

    with pytest.raises(ValueError, match=r"input.fa:1: record 'nsp1': region 3-99"):
        parse_sequences([], [fasta])


def test_record_carries_its_region_and_the_whole_sequence(tmp_path: Path) -> None:
    """The coordinates reach the record; the sequence is never truncated."""
    fasta = tmp_path / "input.fa"
    fasta.write_text('>nsp1|2|4|{"strain": "x"}\nACDEF\n')

    record = parse_sequences([], [fasta])[0]

    assert (record.name, record.start, record.stop) == ("nsp1", 2, 4)
    assert record.sequence == "ACDEF"
    assert record.metadata == {"strain": "x"}


def test_positional_sequences_cover_themselves() -> None:
    """A bare sequence on the command line has no parent to be a region of."""
    record = parse_sequences(["ACDEF"], [])[0]

    assert (record.name, record.start, record.stop) == ("seq_01", 1, 5)
