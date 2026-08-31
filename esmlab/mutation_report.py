"""Report stage: turn stored logits into mutation-analysis artifacts.

Reads a :class:`~esmlab.storage.LogitsStorage` and never constructs a
connector, so this stage needs no credentials, no GPU and no model. Re-running
it with a different threshold or top-k is pure CPU work over arrays that are
already on disk.

Reports are located by the sequence's storage key, not by its label: two input
records sharing a FASTA header are the same analysis if they are the same
sequence. ``manifest.csv`` maps each key back to the label for humans.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from esmlab.amino_acids import VALID_AMINO_ACIDS
from esmlab.connectors.base import SequenceLogits
from esmlab.mutation_scoring import (
    deleterious_fraction_per_position,
    entropy_per_position,
    llr_matrix,
    rank_substitutions,
    tolerant_positions,
)
from esmlab.plotting import plot_deleterious_fraction, plot_entropy, plot_llr_heatmap
from esmlab.storage import FileLogitsStorage, StoredLogits, logits_key

SUMMARY_COLUMNS = (
    "position",
    "wt_aa",
    "entropy_bits",
    "fraction_deleterious",
    "tolerant",
    "top_alt",
    "top_alt_llr",
)

MANIFEST_COLUMNS = ("key", "label", "length", "created_utc")
MANIFEST_FILE = "manifest.csv"


@dataclass(frozen=True)
class ReportSettings:
    """Fully-resolved configuration for one :func:`run_report` invocation.

    Built by the ``mutation_report`` script. ``model`` selects which stored
    entries to report on; ``threshold`` and ``top_k`` are pure presentation
    knobs, free to change without touching the model.
    """

    model: str
    storage_root: Path
    out_dir: Path
    threshold: float
    top_k: int


def run_report(settings: ReportSettings) -> list[Path]:
    """Writes plots, a per-position CSV and a manifest for every stored entry.

    Iterates the storage rather than an input file: the store is the source of
    truth for what has been computed. Each entry lands in
    ``<out_dir>/<model>/<key>/``; the model is part of the path because the
    storage key is the sequence's alone, so two models would otherwise
    overwrite each other. Returns the list of written artifact paths.
    """
    storage = FileLogitsStorage(settings.storage_root)
    model_dir = settings.out_dir / settings.model
    model_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[Path] = []
    manifest_rows: list[tuple[str, str, int, str]] = []
    for entry in storage.entries(model=settings.model):
        key = logits_key(entry.logits.sequence)
        artifacts.extend(_write_entry_report(entry, model_dir / key, settings))
        manifest_rows.append(
            (key, entry.label, len(entry.logits.sequence), entry.created_utc)
        )

    artifacts.append(_write_manifest(model_dir / MANIFEST_FILE, manifest_rows))
    return artifacts


def _write_entry_report(
    entry: StoredLogits, report_dir: Path, settings: ReportSettings
) -> list[Path]:
    """Derives the scores for one entry and writes its plots and summary CSV."""
    result = entry.logits
    entropies = entropy_per_position(result)
    llr = llr_matrix(result)
    fractions = deleterious_fraction_per_position(llr)

    report_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [
        plot_entropy(entropies, result.sequence, report_dir / "entropy.png"),
        plot_deleterious_fraction(
            fractions, settings.threshold, report_dir / "deleterious_fraction.png"
        ),
        plot_llr_heatmap(llr, result.sequence, report_dir / "llr_heatmap.png"),
    ]

    summary_path = report_dir / "summary.csv"
    summary_path.write_text(
        "\n".join(_summary_lines(result, entropies, fractions, llr, settings.threshold))
        + "\n"
    )
    artifacts.append(summary_path)

    _print_report(entry, entropies, fractions, llr, settings)
    return artifacts


def _write_manifest(path: Path, rows: list[tuple[str, str, int, str]]) -> Path:
    """Writes the key-to-label index for the reported model.

    Rewritten wholesale on every run rather than appended, so it always
    describes exactly the entries currently in storage.
    """
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(MANIFEST_COLUMNS)
        writer.writerows(rows)
    return path


def _summary_lines(
    result: SequenceLogits,
    entropies: npt.NDArray[np.float64],
    fractions: npt.NDArray[np.float64],
    llr: npt.NDArray[np.float64],
    threshold: float,
) -> list[str]:
    """Builds the rows of the per-sequence ``summary.csv`` as strings.

    One row per position: 1-indexed position, wildtype, entropy, deleterious
    fraction, tolerant flag, best alternative, and its LLR.
    """
    lines = [",".join(SUMMARY_COLUMNS)]
    for position, wildtype in enumerate(result.sequence):
        alt_column = int(np.argmax(llr[position]))
        lines.append(
            ",".join(
                (
                    str(position + 1),
                    wildtype,
                    f"{entropies[position]:.4f}",
                    f"{fractions[position]:.4f}",
                    str(bool(fractions[position] < threshold)),
                    VALID_AMINO_ACIDS[alt_column],
                    f"{llr[position, alt_column]:+.4f}",
                )
            )
        )
    return lines


def _print_top_positions(
    label: str, entries: list[tuple[int, str, float]], value_format: str
) -> None:
    """Prints one labeled line listing positions as ``A12 (0.123)`` tuples."""
    rendered = ", ".join(
        f"{aa}{position} ({value:{value_format}})" for position, aa, value in entries
    )
    print(f"{label}: {rendered}")


def _print_report(
    entry: StoredLogits,
    entropies: npt.NDArray[np.float64],
    fractions: npt.NDArray[np.float64],
    llr: npt.NDArray[np.float64],
    settings: ReportSettings,
) -> None:
    """Prints the human-readable per-sequence report to stdout.

    The header shows the stored label, which is the only place a display name
    is used. Pulls the most-constrained/most-tolerant positions from
    ``entropies`` and ``fractions``, the top tolerated substitutions from
    :func:`rank_substitutions`, and the tolerant-position set from
    :func:`tolerant_positions`.
    """
    result = entry.logits
    print(f"\n=== {entry.label} (L={len(result.sequence)}, model={entry.model}) ===")
    constrained = sorted(
        zip(range(len(entropies)), result.sequence, entropies),
        key=lambda item: item[2],
    )[:5]
    tolerant = sorted(
        zip(range(len(fractions)), result.sequence, fractions),
        key=lambda item: item[2],
    )[:5]
    _print_top_positions("Most constrained positions", constrained, ".3f")
    _print_top_positions("Most mutation-tolerant positions", tolerant, ".3f")

    substitutions = rank_substitutions(llr, result.sequence, settings.top_k)
    positive = [substitution for substitution in substitutions if substitution[3] > 0]
    print(f"Top {len(positive)} tolerated substitutions (LLR > 0):")
    for position, wildtype, alternative, score in positive:
        print(f"  {wildtype}{position}{alternative}: {score:+.3f} nats")

    selected = tolerant_positions(fractions, settings.threshold)
    print(
        f"{selected.size} position(s) below deleterious-fraction threshold "
        f"{settings.threshold}: {[int(position) + 1 for position in selected[:20]]}"
    )
