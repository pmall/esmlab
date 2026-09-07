"""Sequence input: parsing, validation, and the record it produces.

``NamedSequence`` is the unit of work flowing from the command line into
:mod:`esmlab.inference`; logits persistence lives in :mod:`esmlab.storage`.
Sequences arrive as positional command-line arguments or from FASTA files.

FASTA header formats
--------------------
There are two, one per entry point, and a file is read under one of them.

:func:`parse_sequences` requires coordinates - ``>label|start|stop``, with an
optional JSON object appended::

    >nsp1|60|74|{"source": "UniProt:P12345", "strain": "..."}
    MKT...                                  (the whole protein)
    >stat1|1|15
    VKDKVMCIEHEIKSL

``start`` and ``stop`` are 1-based inclusive residue coordinates naming the
sub-sequence of interest: an entry says which residues it is about. They narrow
what is masked, not what the model reads. The whole sequence goes into every
forward pass, because the surrounding residues are the context that makes the
region's logits mean anything, but only the region's residues are masked and
scored - so a 15-residue peptide inside a 500-residue protein costs 15 forward
passes, not 500.

:func:`parse_whole_sequences` reads a plain FASTA, where the header is a label
and an optional JSON object and nothing else::

    >spike|{"source": "UniProt:P0DTC2"}
    MFVFLVLLPLVSSQ...

A record that names no part of its sequence is about all of it, so the region
is 1..len. That topic reads the model in one unmasked pass per record rather
than one per residue, which is what makes a whole protein affordable; see
:data:`~esmlab.connectors.base.SCORING_METHODS`.

The label is sanitized into a display string; the JSON object is carried
through to storage verbatim and nothing here interprets it. Adding information
to a record therefore costs one appended object and no schema change. A
malformed header fails the run with its file and line rather than silently
storing something else. See :func:`parse_header` and :func:`parse_whole_header`
for how the parts are split.

**The label is a display string only** — never an identifier, never a path
component. Storage is keyed by the sequence, so duplicate labels are accepted
and two records with the same sequence are the same analysis.
"""

import json
from collections.abc import Callable
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


def _split_metadata(header: str) -> tuple[str, Metadata]:
    """Splits the optional ``|{...}`` object off a header, decoding it.

    The split is on the first ``|{`` rather than on ``|`` alone, so a brace in
    a value cannot confuse it, and both header formats therefore carry
    metadata the same way. Malformed JSON raises :class:`ValueError`, so a typo
    fails the run rather than silently storing something else.
    """
    label_part, separator, metadata_part = header.partition("|{")
    if not separator:
        return label_part, {}
    try:
        # The brace is the separator's own, so put it back before decoding.
        return label_part, json.loads("{" + metadata_part)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid JSON metadata in header {header!r}: {error}"
        ) from error


def parse_whole_header(header: str, fallback: str) -> tuple[str, Metadata]:
    """Splits a coordinate-free FASTA header into its label and optional JSON.

    The format is ``label`` with an optional ``|{...}`` appended. Everything
    before that object is the label, pipes included, because a record with no
    region has nothing else to say: an accession like ``sp|P12345|NAME`` is one
    label rather than a malformed coordinate pair.
    """
    label_part, metadata = _split_metadata(header)
    return sanitize_name(label_part, fallback), metadata


def parse_header(header: str, fallback: str) -> tuple[str, int, int, Metadata]:
    """Splits a FASTA header into its label, coordinates and optional JSON.

    The format is ``label|start|stop`` with an optional ``|{...}`` appended.

    Raises :class:`ValueError` on a missing or non-numeric coordinate, so a
    typo fails the run rather than silently storing something else.
    """
    label_part, metadata = _split_metadata(header)
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


type ParsedHeader = tuple[str, tuple[int, int] | None, Metadata]


def _region_header(header: str, fallback: str) -> ParsedHeader:
    """Adapts :func:`parse_header` to what :func:`_parse_fasta` threads through."""
    label, start, stop, metadata = parse_header(header, fallback)
    return label, (start, stop), metadata


def _whole_header(header: str, fallback: str) -> ParsedHeader:
    """Adapts :func:`parse_whole_header`, whose records have no region of their own.

    ``None`` rather than ``(1, len(sequence))``: the residues below the header
    have not been read yet, so the region is resolved in :func:`_record`.
    """
    label, metadata = parse_whole_header(header, fallback)
    return label, None, metadata


def _parse_fasta(
    path: Path, header_parser: Callable[[str, str], ParsedHeader]
) -> list[NamedSequence]:
    """Parses a single FASTA file into raw :class:`NamedSequence` records.

    Headers (``>...``) open a record and are split by ``header_parser`` into a
    sanitized label, an optional region and the optional metadata; sequence
    lines accumulate until the next header. The parser is a parameter because
    the two header formats differ in nothing else. Sequences are not validated
    here — the calling entry point validates every record after merging
    sources. Raises on sequence-before-header, a malformed header, or no
    records found.
    """
    records: list[NamedSequence] = []
    header = ""
    header_line = 0
    region: tuple[int, int] | None = None
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
                header, region, metadata = header_parser(stripped[1:], fallback)
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
    region: tuple[int, int] | None,
    sequence: str,
    metadata: Metadata,
) -> NamedSequence:
    """Closes one FASTA record, resolving and checking its region.

    A record that declared no region covers its whole sequence. Both the
    resolution and the check wait until here because neither can happen until
    the residues below the header have been read; the failure still names the
    header's own line.
    """
    start, stop = region if region is not None else (1, len(sequence))
    try:
        check_region(sequence, start, stop, label)
    except ValueError as error:
        raise ValueError(f"{path}:{line_number}: {error}") from error
    return NamedSequence(label, sequence, start, stop, metadata)


def parse_sequences(
    raw_sequences: list[str], fasta_paths: list[Path]
) -> list[NamedSequence]:
    """Validates positional sequences and coordinate-carrying FASTA files.

    FASTA records keep their sanitized headers, their declared coordinates and
    any JSON the header carried; a header without coordinates is an error, so
    a plain FASTA has to be read by :func:`parse_whole_sequences` instead. The
    returned list is what :func:`~esmlab.inference.run_inference` iterates
    over.
    """
    return _collect(raw_sequences, fasta_paths, _region_header)


def parse_whole_sequences(
    raw_sequences: list[str], fasta_paths: list[Path]
) -> list[NamedSequence]:
    """Validates positional sequences and coordinate-free FASTA files.

    Every record covers its whole sequence, which is the same region a bare
    positional sequence gets, so the two sources agree here in a way they do
    not under :func:`parse_sequences`. The records are otherwise identical, and
    :func:`~esmlab.inference.run_inference` cannot tell which entry point built
    them.
    """
    return _collect(raw_sequences, fasta_paths, _whole_header)


def _collect(
    raw_sequences: list[str],
    fasta_paths: list[Path],
    header_parser: Callable[[str, str], ParsedHeader],
) -> list[NamedSequence]:
    """Merges both input sources into one validated list, under one header format.

    Positional sequences are labeled ``seq_01``, ``seq_02``, ... , carry no
    metadata, and cover themselves entirely - a bare sequence on the command
    line has nothing around it to be a region of - so they read the same way
    whichever entry point was called. Every record is validated by
    :func:`validate_sequence`. Duplicate labels are accepted because labels are
    display-only — storage is keyed by the sequence and its region.
    """
    sequences: list[NamedSequence] = []
    for index, raw in enumerate(raw_sequences, start=1):
        sequence = validate_sequence(raw)
        sequences.append(NamedSequence(f"seq_{index:02d}", sequence, 1, len(sequence)))
    for fasta_path in fasta_paths:
        for record in _parse_fasta(fasta_path, header_parser):
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
