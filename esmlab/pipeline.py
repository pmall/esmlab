"""Orchestration of the unified mutation-analysis CLI stage.

Inference and reporting run in one pass: the connector (cache-decorated in
``get_connector``) serves cached logits on hit and recomputes on miss, then
the pure-CPU analysis derives entropy, LLR, and the report artifacts.
"""

import csv
import resource
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

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

PERF_COLUMNS = (
    "started_utc",
    "backend",
    "model",
    "n_sequences",
    "total_residues",
    "init_duration_s",
    "inference_duration_s",
    "mean_ms_per_residue",
    "cache_hits",
    "cache_misses",
    "process_peak_rss_bytes",
    "gpu_peak_bytes",
    "out_dir",
)


@dataclass(frozen=True)
class AnalysisSettings:
    """Fully-resolved configuration for one CLI invocation of :func:`run_analysis`.

    Bundles the selected backend/model and its resolved parameters, the
    parsed input sequences, output directory, the scoring/report knobs
    (threshold, top_k, cache), and the longitudinal perf-report CSV path.
    Built by the script's :func:`_analyze`.
    """

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
    perf_report: Path


@dataclass(frozen=True)
class _SeqTiming:
    """One sequence's performance record, collected during :func:`run_analysis`.

    ``gpu_peak_bytes`` is ``None`` when the backend cannot observe memory
    (stub, forge, modal, or the local backend on CPU).
    """

    name: str
    length: int
    duration_s: float
    cache_hit: bool
    gpu_peak_bytes: int | None


def _format_bytes(num_bytes: int | None) -> str:
    """Renders a byte count as a human-readable string, or ``n/a`` for ``None``."""
    if num_bytes is None:
        return "n/a"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.0f} {unit}"
        num_bytes = int(num_bytes / 1024)
    return f"{num_bytes:.0f} PB"


def _print_top_positions(
    label: str, entries: list[tuple[int, str, float]], value_format: str
) -> None:
    """Prints one labeled line listing positions as ``A12 (0.123)`` tuples.

    Uses :func:`tqdm.write` so output does not clobber an active progress bar.
    Used by :func:`_print_report` to render the most-constrained and
    most-tolerant position summaries.
    """
    rendered = ", ".join(
        f"{aa}{position} ({value:{value_format}})" for position, aa, value in entries
    )
    tqdm.write(f"{label}: {rendered}")


def _print_report(
    result: SequenceLogits,
    meta: InferenceMeta,
    entropies: npt.NDArray[np.float64],
    fractions: npt.NDArray[np.float64],
    llr: npt.NDArray[np.float64],
    threshold: float,
    top_k: int,
) -> None:
    """Prints the human-readable per-sequence report to stdout.

    All output goes through :func:`tqdm.write` so it layers above an active
    progress bar without breaking it. Pulls the most-constrained/most-tolerant
    positions from ``entropies`` and ``fractions``, the top tolerated
    substitutions from :func:`rank_substitutions`, and the tolerant-position
    set from :func:`tolerant_positions`. Called once per sequence by
    :func:`run_analysis`.
    """
    tqdm.write(
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
    tqdm.write(f"Top {len(positive)} tolerated substitutions (LLR > 0):")
    for position, wildtype, alternative, score in positive:
        tqdm.write(f"  {wildtype}{position}{alternative}: {score:+.3f} nats")

    selected = tolerant_positions(fractions, threshold)
    tqdm.write(
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
    """Builds the rows of the per-sequence ``summary.csv`` as strings.

    One row per position: 1-indexed position, wildtype, entropy, deleterious
    fraction, tolerant flag, best alternative, and its LLR. Called by
    :func:`run_analysis`, which joins the lines and writes the file.
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


@dataclass(frozen=True)
class _RunAgg:
    """Aggregate of one run's per-sequence timings, shared by the two report sinks."""

    n_sequences: int
    total_residues: int
    inference_duration: float
    mean_ms_per_residue: float
    hits: int
    misses: int
    gpu_peak_bytes: int | None


def _aggregate(timings: list[_SeqTiming]) -> _RunAgg:
    """Reduces per-sequence timings to the run-level metrics both report sinks need."""
    total_residues = sum(timing.length for timing in timings)
    inference_duration = sum(timing.duration_s for timing in timings)
    mean_ms_per_residue = (
        inference_duration / total_residues * 1000 if total_residues else 0.0
    )
    hits = sum(1 for timing in timings if timing.cache_hit)
    misses = len(timings) - hits
    gpu_peaks = [
        timing.gpu_peak_bytes for timing in timings if timing.gpu_peak_bytes is not None
    ]
    return _RunAgg(
        n_sequences=len(timings),
        total_residues=total_residues,
        inference_duration=inference_duration,
        mean_ms_per_residue=mean_ms_per_residue,
        hits=hits,
        misses=misses,
        gpu_peak_bytes=max(gpu_peaks) if gpu_peaks else None,
    )


def _print_perf_summary(
    settings: AnalysisSettings,
    init_duration: float,
    agg: _RunAgg,
    process_peak_rss: int,
) -> None:
    """Prints the end-of-run performance summary to stdout.

    Called after the progress bar closes. Reports sequence/residue counts,
    init and inference durations, mean per-residue latency, cache hit/miss
    counts, and the peak process RSS plus the max GPU peak across sequences.
    """
    print(
        f"\n=== performance ({settings.backend}/{settings.model}) ===\n"
        f"sequences: {agg.n_sequences} ({agg.total_residues} residues)\n"
        f"init: {init_duration:.3f}s, inference: {agg.inference_duration:.3f}s, "
        f"mean: {agg.mean_ms_per_residue:.3f} ms/residue\n"
        f"cache: {agg.hits} hit / {agg.misses} miss\n"
        f"process RSS peak: {_format_bytes(process_peak_rss)}\n"
        f"GPU peak: {_format_bytes(agg.gpu_peak_bytes)}\n"
        f"perf report: {settings.perf_report}"
    )


def _append_perf_csv(
    settings: AnalysisSettings,
    started_utc: str,
    init_duration: float,
    agg: _RunAgg,
    process_peak_rss: int,
) -> None:
    """Appends one aggregate row to the longitudinal perf-report CSV.

    One row per :func:`run_analysis` invocation; the header is written only
    when the file is first created. GPU peak is the max across sequences
    (empty when no backend observed memory).
    """
    gpu_peak = "" if agg.gpu_peak_bytes is None else str(agg.gpu_peak_bytes)
    row = [
        started_utc,
        settings.backend,
        settings.model,
        str(agg.n_sequences),
        str(agg.total_residues),
        f"{init_duration:.6f}",
        f"{agg.inference_duration:.6f}",
        f"{agg.mean_ms_per_residue:.6f}",
        str(agg.hits),
        str(agg.misses),
        str(process_peak_rss),
        gpu_peak,
        str(settings.out_dir),
    ]
    is_new = not settings.perf_report.exists()
    settings.perf_report.parent.mkdir(parents=True, exist_ok=True)
    with settings.perf_report.open("a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if is_new:
            writer.writerow(PERF_COLUMNS)
        writer.writerow(row)


def run_analysis(settings: AnalysisSettings) -> list[Path]:
    """Scores each sequence (cache-aware) and writes plots, CSV, and a console report.

    Builds the cache-decorated connector via :func:`get_connector`, then for
    each :class:`NamedSequence` runs the connector, derives entropy/LLR/
    deleterious fraction via :mod:`esmlab.mutation_scoring`, writes the three
    plots via :mod:`esmlab.plotting`, writes ``summary.csv`` via
    :func:`_summary_lines`, and prints the report via :func:`_print_report`.
    A :class:`tqdm` progress bar tracks sequence-level progress; on completion
    a perf summary is printed and one row appended to the perf-report CSV.
    Entrypoint called by the script's :func:`_analyze`; returns the list of
    written per-sequence artifact paths (the perf CSV is run-level and not
    included in this list).
    """
    started_utc = datetime.now(UTC).isoformat()
    init_start = perf_counter()
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
    init_duration = perf_counter() - init_start

    timings: list[_SeqTiming] = []
    artifacts: list[Path] = []
    with tqdm(settings.sequences, unit="seq", desc="analysis") as progress:
        for named in progress:
            progress.set_postfix_str(f"{named.name} L={len(named.sequence)}")
            infer_start = perf_counter()
            result = connector.masked_sequence_logits(named.sequence)
            duration = perf_counter() - infer_start
            timing = _SeqTiming(
                name=named.name,
                length=len(named.sequence),
                duration_s=duration,
                cache_hit=connector.last_cache_hit,
                gpu_peak_bytes=connector.peak_memory_bytes(),
            )
            timings.append(timing)
            progress.set_postfix_str(
                f"{named.name} {duration:.2f}s "
                f"hit={'yes' if timing.cache_hit else 'no'}"
            )

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
                    fractions,
                    settings.threshold,
                    report_dir / "deleterious_fraction.png",
                )
            )
            artifacts.append(
                plot_llr_heatmap(llr, result.sequence, report_dir / "llr_heatmap.png")
            )

            summary_path = report_dir / "summary.csv"
            summary_path.write_text(
                "\n".join(
                    _summary_lines(
                        result, entropies, fractions, llr, settings.threshold
                    )
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
                result,
                meta,
                entropies,
                fractions,
                llr,
                settings.threshold,
                settings.top_k,
            )

    process_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    agg = _aggregate(timings)
    _print_perf_summary(settings, init_duration, agg, process_peak_rss)
    _append_perf_csv(settings, started_utc, init_duration, agg, process_peak_rss)
    return artifacts
