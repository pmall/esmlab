"""Sequence input: parsing, validation, and the record it produces.

``NamedSequence`` is the unit of work flowing from the command line into
:mod:`esmlab.inference`; logits persistence lives in :mod:`esmlab.storage`.
Sequences arrive as positional command-line arguments or from FASTA files.

FASTA header format
-------------------
A header is ``>label`` or ``>label|{...}``, the JSON object optional::

    >stat1|{"source": "UniProt:P12345", "targets": ["P11111", "P22222"]}
    VKDKVMCIEHEIKSL
    >stat2
    MKTAYIAKQRQISFVK

The label is sanitized into a display string; the object is carried through to
storage verbatim and nothing here interprets it. Adding information to a record
therefore costs one appended object and no schema change. Malformed JSON fails
the run with its file and line rather than silently storing nothing. See
:func:`parse_header` for how the two halves are split.

**The label is a display string only** — never an identifier, never a path
component. Storage is keyed by the sequence, so duplicate labels are accepted
and two records with the same sequence are the same analysis.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from esmlab.amino_acids import VALID_AMINO_ACIDS
from esmlab.storage import Metadata


@dataclass(frozen=True)
class NamedSequence:
    """A protein sequence with a display label and the record it came from.

    The label is the sanitized FASTA header, carried purely so runs and reports
    are readable. It is never an identifier: the sequence alone keys storage,
    so two records may share a label without conflict.

    ``metadata`` is the optional JSON object from the record's header, carried
    through to storage untouched. Its shape is the caller's business: the
    parser only decodes it.
    """

    name: str
    sequence: str
    metadata: Metadata = field(default_factory=dict)


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


def parse_header(header: str, fallback: str) -> tuple[str, Metadata]:
    """Splits a FASTA header into its display label and optional JSON metadata.

    The format is ``label`` or ``label|{...}``, so adding a bit more
    information to a record costs one appended object and nothing else changes.
    The split is on the first ``|{`` rather than on ``|`` alone, which leaves
    conventional pipe-separated identifiers (``sp|P12345|NAME``) whole: those
    never precede a brace. A header with no ``|{`` yields an empty object.

    Raises :class:`ValueError` on malformed JSON so a typo fails the run rather
    than silently storing nothing.
    """
    label_part, separator, metadata_part = header.partition("|{")
    label = sanitize_name(label_part, fallback)
    if not separator:
        return label, {}
    try:
        # The brace is the separator's own, so put it back before decoding.
        return label, json.loads("{" + metadata_part)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON metadata in header {header!r}: {error}") from (
            error
        )


def _parse_fasta(path: Path) -> list[NamedSequence]:
    """Parses a single FASTA file into raw :class:`NamedSequence` records.

    Headers (``>...``) open a record and are split by :func:`parse_header` into
    a sanitized label and its optional metadata; sequence lines accumulate
    until the next header. Sequences are not validated here —
    :func:`parse_sequences` validates every record after merging sources.
    Raises on sequence-before-header, malformed header JSON, or no records
    found.
    """
    records: list[NamedSequence] = []
    header = ""
    metadata: Metadata = {}
    chunks: list[str] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith(">"):
            if header:
                records.append(NamedSequence(header, "".join(chunks), metadata))
            fallback = f"{path.stem}_{len(records) + 1}"
            try:
                header, metadata = parse_header(stripped[1:], fallback)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            chunks = []
        elif stripped:
            if not header:
                raise ValueError(
                    f"{path}:{line_number}: sequence before any '>' header"
                )
            chunks.append(stripped)
    if header:
        records.append(NamedSequence(header, "".join(chunks), metadata))
    if not records:
        raise ValueError(f"{path}: no FASTA records found")
    return records


def parse_sequences(
    raw_sequences: list[str], fasta_paths: list[Path]
) -> list[NamedSequence]:
    """Validates positional sequences and FASTA files into one labeled list.

    Positional sequences are labeled ``seq_01``, ``seq_02``, ... and carry no
    metadata; FASTA records keep their sanitized headers and any JSON the
    header declared. Every record is validated by
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
                NamedSequence(
                    record.name, validate_sequence(record.sequence), record.metadata
                )
            )
    if not sequences:
        raise ValueError("No input sequences given")
    return sequences
