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
    name: str
    sequence: str


@dataclass(frozen=True)
class InferenceMeta:
    name: str
    backend: str
    model: str
    created_utc: str


def validate_sequence(sequence: str) -> str:
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
    cleaned = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in candidate.strip()
    ).strip("_")
    return cleaned or fallback


def _parse_fasta(path: Path) -> list[NamedSequence]:
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
    """Validates positional sequences and FASTA files into one named list."""
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
