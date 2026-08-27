"""Sequence parsing and the shared domain types for inference artifacts.

``NamedSequence``/``InferenceMeta`` are the lightweight records flowing between
parsing, the cache, and the report; logits persistence itself lives in
:mod:`esmlab.cache`.
"""

from dataclasses import dataclass
from pathlib import Path

from esmlab.amino_acids import VALID_AMINO_ACIDS


@dataclass(frozen=True)
class NamedSequence:
    """A protein sequence paired with its report name.

    The unit of work flowing through the pipeline: :func:`parse_sequences`
    produces them, :func:`run_analysis` consumes them, and the name becomes
    the per-sequence output subdirectory.
    """

    name: str
    sequence: str


@dataclass(frozen=True)
class InferenceMeta:
    """Provenance stamped onto a console report (name, backend, model, time)."""

    name: str
    backend: str
    model: str
    created_utc: str


def validate_sequence(sequence: str) -> str:
    """Normalizes and validates one raw sequence against the canonical amino acids.

    Strips whitespace, uppercases, and rejects empty input or any non-canonical
    residue. Called by :func:`parse_sequences` for both positional and FASTA
    inputs so downstream code only ever sees clean sequences.
    """
    cleaned = sequence.strip().upper()
    invalid = sorted(set(cleaned) - set(VALID_AMINO_ACIDS))
    if not cleaned:
        raise ValueError("Empty sequence")
    if invalid:
        raise ValueError(
            f"Sequence contains non-canonical residues: {''.join(invalid)}"
        )
    return cleaned


def sanitize_name(candidate: str, fallback: str) -> str:
    """Turns an arbitrary FASTA header into a filesystem-safe report name.

    Non-alphanumeric characters (except ``-_.``) become ``_`` and surrounding
    underscores are stripped; an empty result falls back to ``fallback``. Used
    by :func:`_parse_fasta` so each record's name can serve as an output
    directory.
    """
    cleaned = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in candidate.strip()
    ).strip("_")
    return cleaned or fallback


def _parse_fasta(path: Path) -> list[NamedSequence]:
    """Parses a single FASTA file into raw :class:`NamedSequence` records.

    Headers (``>...``) open a record and are sanitized via
    :func:`sanitize_name`; sequence lines accumulate until the next header.
    Sequences are not validated here — :func:`parse_sequences` validates every
    record after merging sources. Raises on sequence-before-header or no
    records found.
    """
    records: list[NamedSequence] = []
    header = ""
    chunks: list[str] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith(">"):
            if header:
                records.append(NamedSequence(header, "".join(chunks)))
            header = sanitize_name(stripped[1:], f"{path.stem}_{len(records) + 1}")
            chunks = []
        elif stripped:
            if not header:
                raise ValueError(
                    f"{path}:{line_number}: sequence before any '>' header"
                )
            chunks.append(stripped)
    if header:
        records.append(NamedSequence(header, "".join(chunks)))
    if not records:
        raise ValueError(f"{path}: no FASTA records found")
    return records


def parse_sequences(
    raw_sequences: list[str], fasta_paths: list[Path]
) -> list[NamedSequence]:
    """Validates positional sequences and FASTA files into one named list.

    Positional sequences are named ``seq_01``, ``seq_02``, ...; FASTA records
    keep their sanitized headers. Every record is validated by
    :func:`validate_sequence`, duplicate names and empty input are rejected.
    The returned list is what :func:`run_analysis` iterates over.
    """
    sequences: list[NamedSequence] = []
    for index, raw in enumerate(raw_sequences, start=1):
        sequences.append(NamedSequence(f"seq_{index:02d}", validate_sequence(raw)))
    for fasta_path in fasta_paths:
        for record in _parse_fasta(fasta_path):
            sequences.append(
                NamedSequence(record.name, validate_sequence(record.sequence))
            )
    names = [named.name for named in sequences]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate sequence names: {', '.join(duplicates)}")
    if not sequences:
        raise ValueError("No input sequences given")
    return sequences
