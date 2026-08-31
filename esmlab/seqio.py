"""Sequence parsing and the input record it produces.

``NamedSequence`` is the unit of work flowing from the command line into
:mod:`esmlab.inference`; logits persistence lives in :mod:`esmlab.storage`.
"""

from dataclasses import dataclass
from pathlib import Path

from esmlab.amino_acids import VALID_AMINO_ACIDS


@dataclass(frozen=True)
class NamedSequence:
    """A protein sequence paired with a display label.

    The label is the FASTA header the sequence arrived under, carried purely
    so runs and reports are readable. It is never an identifier: the sequence
    alone keys storage, so two records may share a label without conflict.
    """

    name: str
    sequence: str


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
    """Turns an arbitrary FASTA header into a safe display label.

    Non-alphanumeric characters (except ``-_.``) become ``_`` and surrounding
    underscores are stripped; an empty result falls back to ``fallback``. Used
    by :func:`_parse_fasta` so a label is safe in filenames and CSV cells.
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
    """Validates positional sequences and FASTA files into one labeled list.

    Positional sequences are labeled ``seq_01``, ``seq_02``, ...; FASTA records
    keep their sanitized headers. Every record is validated by
    :func:`validate_sequence`. Duplicate labels are accepted because labels are
    display-only — storage is keyed by the sequence. The returned list is what
    :func:`~esmlab.inference.run_inference` iterates over.
    """
    sequences: list[NamedSequence] = []
    for index, raw in enumerate(raw_sequences, start=1):
        sequences.append(NamedSequence(f"seq_{index:02d}", validate_sequence(raw)))
    for fasta_path in fasta_paths:
        for record in _parse_fasta(fasta_path):
            sequences.append(
                NamedSequence(record.name, validate_sequence(record.sequence))
            )
    if not sequences:
        raise ValueError("No input sequences given")
    return sequences
