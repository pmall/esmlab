"""Sequence input: parsing, validation, and the record it produces.

``NamedSequence`` is the unit of work flowing from the command line into
:mod:`esmlab.inference`; logits persistence lives in :mod:`esmlab.storage`.
Sequences arrive as positional command-line arguments or from FASTA files.

FASTA header format
-------------------
Every header is ``>label|start|stop``, with an optional JSON object appended::

    >nsp1|60|74|{"source": "UniProt:P12345", "strain": "..."}
    MKT...                                  (the whole protein)
    >stat1|1|15
    VKDKVMCIEHEIKSL

``start`` and ``stop`` are 1-based inclusive residue coordinates naming the
sub-sequence of interest, and they are required: an entry says which residues
it is about. They narrow what is masked, not what the model reads. The whole
sequence goes into every forward pass, because the surrounding residues are the
context that makes the region's logits mean anything, but only the region's
residues are masked and scored - so a 15-residue peptide inside a 500-residue
protein costs 15 forward passes, not 500.

The label is sanitized into a display string; the JSON object is carried
through to storage verbatim and nothing here interprets it. Adding information
to a record therefore costs one appended object and no schema change. A
malformed header fails the run with its file and line rather than silently
storing something else. See :func:`parse_header` for how the parts are split.

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

    ``start`` and ``stop`` are the record's 1-based inclusive coordinates: the
    sub-sequence to mask and report on. The sequence itself is always carried
    whole, because it is the context the region is scored in - and the same
    residues taken from two different sequences are two different records.

    ``metadata`` is the optional JSON object from the record's header, carried
    through to storage untouched. Its shape is the caller's business: the
    parser only decodes it.
    """

    name: str
    sequence: str
    start: int
    stop: int
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


def parse_header(header: str, fallback: str) -> tuple[str, int, int, Metadata]:
    """Splits a FASTA header into its label, coordinates and optional JSON.

    The format is ``label|start|stop`` with an optional ``|{...}`` appended.
    The JSON split is on the first ``|{`` rather than on ``|`` alone, so a
    brace in a value cannot confuse it.

    Raises :class:`ValueError` on a missing or non-numeric coordinate and on
    malformed JSON, so a typo fails the run rather than silently storing
    something else.
    """
    label_part, separator, metadata_part = header.partition("|{")
    metadata: Metadata = {}
    if separator:
        try:
            # The brace is the separator's own, so put it back before decoding.
            metadata = json.loads("{" + metadata_part)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSON metadata in header {header!r}: {error}"
            ) from error

    fields = label_part.rstrip("|").split("|")
    if len(fields) < 3 or not (fields[-2].isdigit() and fields[-1].isdigit()):
        raise ValueError(
            f"header {header!r} must be 'label|start|stop' with optional |{{...}}"
        )
    label = sanitize_name("|".join(fields[:-2]), fallback)
    return label, int(fields[-2]), int(fields[-1]), metadata


def check_region(sequence: str, start: int, stop: int, label: str) -> None:
    """Checks a header's coordinates against the sequence it was declared on.

    Coordinates are 1-based and inclusive, so a valid region satisfies
    ``1 <= start <= stop <= len(sequence)``. A region pointing past its own
    sequence is a typo in the header, and silently clamping it would score
    residues the author did not mean.
    """
    if not 1 <= start <= stop <= len(sequence):
        raise ValueError(
            f"record {label!r}: region {start}-{stop} is not within "
            f"1-{len(sequence)} (coordinates are 1-based and inclusive)"
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
    header_line = 0
    region = (0, 0)
    metadata: Metadata = {}
    chunks: list[str] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith(">"):
            if header:
                records.append(
                    _record(
                        path, header_line, header, region, "".join(chunks), metadata
                    )
                )
            fallback = f"{path.stem}_{len(records) + 1}"
            try:
                header, start, stop, metadata = parse_header(stripped[1:], fallback)
                region = (start, stop)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            header_line = line_number
            chunks = []
        elif stripped:
            if not header:
                raise ValueError(
                    f"{path}:{line_number}: sequence before any '>' header"
                )
            chunks.append(stripped)
    if header:
        records.append(
            _record(path, header_line, header, region, "".join(chunks), metadata)
        )
    if not records:
        raise ValueError(f"{path}: no FASTA records found")
    return records


def _record(
    path: Path,
    line_number: int,
    label: str,
    region: tuple[int, int],
    sequence: str,
    metadata: Metadata,
) -> NamedSequence:
    """Closes one FASTA record, checking its region against the sequence read.

    The check waits until here because a header's coordinates cannot be
    validated until the residues below it have been read; the failure still
    names the header's own line.
    """
    start, stop = region
    try:
        check_region(sequence, start, stop, label)
    except ValueError as error:
        raise ValueError(f"{path}:{line_number}: {error}") from error
    return NamedSequence(label, sequence, start, stop, metadata)


def parse_sequences(
    raw_sequences: list[str], fasta_paths: list[Path]
) -> list[NamedSequence]:
    """Validates positional sequences and FASTA files into one labeled list.

    Positional sequences are labeled ``seq_01``, ``seq_02``, ... , carry no
    metadata, and cover themselves entirely - a bare sequence on the command
    line has nothing around it to be a region of. FASTA records keep their
    sanitized headers, their declared coordinates and any JSON the header
    carried. Every record is validated by :func:`validate_sequence`. Duplicate
    labels are accepted because labels are display-only — storage is keyed by
    the sequence and its region. The returned list is what
    :func:`~esmlab.inference.run_inference` iterates over.
    """
    sequences: list[NamedSequence] = []
    for index, raw in enumerate(raw_sequences, start=1):
        sequence = validate_sequence(raw)
        sequences.append(NamedSequence(f"seq_{index:02d}", sequence, 1, len(sequence)))
    for fasta_path in fasta_paths:
        for record in _parse_fasta(fasta_path):
            sequences.append(
                NamedSequence(
                    record.name,
                    validate_sequence(record.sequence),
                    record.start,
                    record.stop,
                    record.metadata,
                )
            )
    if not sequences:
        raise ValueError("No input sequences given")
    return sequences
