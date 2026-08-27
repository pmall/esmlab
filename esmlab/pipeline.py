"""Orchestration of the unified mutation-analysis CLI stage.

Inference and reporting run in one pass: the connector (cache-decorated in
``get_connector``) serves cached logits on hit and recomputes on miss, then
the pure-CPU analysis derives entropy, LLR, and the report artifacts.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt

from esmlab.amino_acids import VALID_AMINO_ACIDS
from esmlab.connectors import get_connector
from esmlab.connectors.base import SequenceLogits
from esmlab.mutation_scoring import (
    deleterious_fraction_per_position,
    entropy_per_position,
    llr_matrix,
    rank_substitutions,
    tolerant_positions,
)
from esmlab.plotting import plot_deleterious_fraction, plot_entropy, plot_llr_heatmap
from esmlab.seqio import InferenceMeta, NamedSequence

SUMMARY_COLUMNS = (
    "position",
    "wt_aa",
    "entropy_bits",
    "fraction_deleterious",
    "tolerant",
    "top_alt",
    "top_alt_llr",
)


@dataclass(frozen=True)
class AnalysisSettings:
    backend: str
    model: str
    device: str
    batch_size: int
    forge_api_key: str
    modal_token_id: str
    modal_token_secret: str
    sequences: list[NamedSequence]
    out_dir: Path
    threshold: float
    top_k: int
    cache: str
    cache_root: Path | None


def _print_top_positions(
    label: str, entries: list[tuple[int, str, float]], value_format: str
) -> None:
    rendered = ", ".join(
        f"{aa}{position} ({value:{value_format}})" for position, aa, value in entries
    )
    print(f"{label}: {rendered}")


def _print_report(
    result: SequenceLogits,
    meta: InferenceMeta,
    entropies: npt.NDArray[np.float64],
    fractions: npt.NDArray[np.float64],
    llr: npt.NDArray[np.float64],
    threshold: float,
    top_k: int,
) -> None:
    print(
        f"\n=== {meta.name} "
        f"(L={len(result.sequence)}, model={meta.model}, backend={meta.backend}) ==="
    )
    constrained = sorted(
        zip(range(len(entropies)), result.sequence, entropies),
        key=lambda entry: entry[2],
    )[:5]
    tolerant = sorted(
        zip(range(len(fractions)), result.sequence, fractions),
        key=lambda entry: entry[2],
    )[:5]
    _print_top_positions("Most constrained positions", constrained, ".3f")
    _print_top_positions("Most mutation-tolerant positions", tolerant, ".3f")

    substitutions = rank_substitutions(llr, result.sequence, top_k)
    positive = [substitution for substitution in substitutions if substitution[3] > 0]
    print(f"Top {len(positive)} tolerated substitutions (LLR > 0):")
    for position, wildtype, alternative, score in positive:
        print(f"  {wildtype}{position}{alternative}: {score:+.3f} nats")

    selected = tolerant_positions(fractions, threshold)
    print(
        f"{selected.size} position(s) below deleterious-fraction threshold {threshold}: "
        f"{[int(position) + 1 for position in selected[:20]]}"
    )


def _summary_lines(
    result: SequenceLogits,
    entropies: npt.NDArray[np.float64],
    fractions: npt.NDArray[np.float64],
    llr: npt.NDArray[np.float64],
    threshold: float,
) -> list[str]:
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


def run_analysis(settings: AnalysisSettings) -> list[Path]:
    """Scores each sequence (cache-aware) and writes plots, CSV, and a console report."""
    connector = get_connector(
        settings.backend,
        settings.model,
        device=settings.device,
        batch_size=settings.batch_size,
        forge_api_key=settings.forge_api_key,
        modal_token_id=settings.modal_token_id,
        modal_token_secret=settings.modal_token_secret,
        cache=settings.cache,
        cache_root=settings.cache_root,
    )
    artifacts: list[Path] = []
    for named in settings.sequences:
        print(
            f"[analyze] {named.name} (L={len(named.sequence)}) "
            f"via {settings.backend}/{settings.model} ..."
        )
        result = connector.masked_sequence_logits(named.sequence)
        entropies = entropy_per_position(result)
        llr = llr_matrix(result)
        fractions = deleterious_fraction_per_position(llr)

        report_dir = settings.out_dir / named.name
        report_dir.mkdir(parents=True, exist_ok=True)
        artifacts.append(
            plot_entropy(entropies, result.sequence, report_dir / "entropy.png")
        )
        artifacts.append(
            plot_deleterious_fraction(
                fractions, settings.threshold, report_dir / "deleterious_fraction.png"
            )
        )
        artifacts.append(
            plot_llr_heatmap(llr, result.sequence, report_dir / "llr_heatmap.png")
        )

        summary_path = report_dir / "summary.csv"
        summary_path.write_text(
            "\n".join(
                _summary_lines(result, entropies, fractions, llr, settings.threshold)
            )
            + "\n"
        )
        artifacts.append(summary_path)

        meta = InferenceMeta(
            name=named.name,
            backend=settings.backend,
            model=settings.model,
            created_utc=datetime.now(UTC).isoformat(),
        )
        _print_report(
            result, meta, entropies, fractions, llr, settings.threshold, settings.top_k
        )
    return artifacts
