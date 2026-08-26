"""Sequence parsing plus read/write of the on-disk inference format.

The intermediate format decouples the costly inference step from reporting:
``infer`` writes one run directory per sequence (logits + metadata JSON),
``report`` reads those directories back without ever touching a backend.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt

from esmlab.amino_acids import VALID_AMINO_ACIDS
from esmlab.connectors.base import SequenceLogits


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


LOGITS_FILE = "logits.npz"
META_FILE = "meta.json"


def save_inference(
    run_dir: Path,
    result: SequenceLogits,
    *,
    backend: str,
    model: str,
) -> Path:
    """Writes one run directory holding the reusable intermediate format."""
    meta = InferenceMeta(
        name=run_dir.name,
        backend=backend,
        model=model,
        created_utc=datetime.now(UTC).isoformat(),
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(run_dir / LOGITS_FILE, logits=result.logits)
    payload = {
        "name": meta.name,
        "backend": meta.backend,
        "model": meta.model,
        "created_utc": meta.created_utc,
        "sequence": result.sequence,
        # Vocab is stored so reports never need to import the tokenizer.
        "vocab": dict(result.vocab),
    }
    (run_dir / META_FILE).write_text(json.dumps(payload))
    return run_dir


def load_inference(run_dir: Path) -> tuple[SequenceLogits, InferenceMeta]:
    payload = json.loads((run_dir / META_FILE).read_text())
    logits: npt.NDArray[np.float32] = np.load(run_dir / LOGITS_FILE)["logits"]
    result = SequenceLogits(
        sequence=payload["sequence"],
        logits=logits,
        vocab=payload["vocab"],
    )
    meta = InferenceMeta(
        name=payload["name"],
        backend=payload["backend"],
        model=payload["model"],
        created_utc=payload["created_utc"],
    )
    return result, meta
