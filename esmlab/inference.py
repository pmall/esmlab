"""Compute stage: run a sequence model over sequences and persist the logits.

Deliberately domain-agnostic. Masked logits are the raw material for mutation
scoring, embedding, layer sweeps and classification alike, so this module knows
nothing about any of them — it orchestrates a connector and a
:class:`~esmlab.storage.LogitsStorage` and stops there.

Persistence is unconditional: an entry already in storage is skipped, never
recomputed. That is the whole point of separating this stage from reporting.

A record is scored only at the residues its header named. The whole sequence
still goes into every forward pass - it is the context that makes the
predictions worth anything - but masking runs over that region alone, so a
15-residue peptide inside a 500-residue protein costs 15 passes rather than 500.
"""

import csv
import resource
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from tqdm import tqdm

from esmlab.connectors import get_connector
from esmlab.connectors.base import ModelConnector
from esmlab.seqio import NamedSequence
from esmlab.storage import LogitsStorage, StorageSettings, logits_key, open_storage

PERF_COLUMNS = (
    "started_utc",
    "backend",
    "model",
    "n_sequences",
    "total_residues",
    "init_duration_s",
    "inference_duration_s",
    "mean_ms_per_residue",
    "computed",
    "skipped",
    "process_peak_rss_bytes",
    "gpu_peak_bytes",
    "storage",
)


@dataclass(frozen=True)
class InferenceSettings:
    """Fully-resolved configuration for one :func:`run_inference` invocation.

    Bundles the selected backend/model with its resolved backend parameters,
    the parsed input sequences, the resolved storage configuration, and the
    longitudinal perf-report CSV path. Built by the ``peptides_logits`` script.
    """

    backend: str
    model: str
    device: str
    batch_size: int
    biohub_api_key: str
    modal_token_id: str
    modal_token_secret: str
    modal_gpu: str
    sequences: list[NamedSequence]
    storage: StorageSettings
    perf_report: Path


@dataclass(frozen=True)
class InferenceStats:
    """What one run did, returned to the caller and mirrored into the perf CSV.

    ``n_sequences`` counts distinct sequences, so a sequence listed twice in
    the input is one unit of work rather than one computed and one skipped.
    """

    n_sequences: int
    computed: int
    skipped: int


@dataclass(frozen=True)
class _SeqTiming:
    """One sequence's performance record, collected during :func:`run_inference`.

    ``key`` identifies the sequence without depending on its display label.
    ``gpu_peak_bytes`` is ``None`` when the backend cannot observe memory
    (stub, biohub, modal, or the local backend on CPU) or when nothing ran.
    """

    key: str
    length: int
    duration_s: float
    computed: bool
    gpu_peak_bytes: int | None


def _distinct_sequences(sequences: list[NamedSequence]) -> list[NamedSequence]:
    """Drops repeated sequences, keeping the first occurrence and its label.

    Two records are the same work only if they name the same region of the same
    sequence, so one protein may appear twice under two regions. Storage would
    already make a true repeat a no-op, but counting it as a "skip" would
    misreport the run: nothing was skipped, the same request was simply listed
    twice.
    """
    seen: dict[tuple[str, int, int], NamedSequence] = {}
    for named in sequences:
        seen.setdefault((named.sequence, named.start, named.stop), named)
    return list(seen.values())


def run_inference(settings: InferenceSettings) -> InferenceStats:
    """Computes and stores logits for every distinct input sequence.

    Builds the connector via :func:`~esmlab.connectors.get_connector`, then for
    each sequence asks storage first and only calls the model on a miss. A
    :class:`tqdm` progress bar tracks sequence-level progress; on completion a
    perf summary is printed and one row appended to the perf-report CSV.
    Entrypoint called by the ``peptides_logits`` script.
    """
    started_utc = datetime.now(UTC).isoformat()
    storage = open_storage(settings.storage)

    init_start = perf_counter()
    connector = get_connector(
        settings.backend,
        settings.model,
        device=settings.device,
        batch_size=settings.batch_size,
        biohub_api_key=settings.biohub_api_key,
        modal_token_id=settings.modal_token_id,
        modal_token_secret=settings.modal_token_secret,
        modal_gpu=settings.modal_gpu,
    )
    init_duration = perf_counter() - init_start

    timings = _compute_missing(settings, connector, storage)

    process_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    agg = _aggregate(timings)
    _print_perf_summary(settings, init_duration, agg, process_peak_rss)
    _append_perf_csv(settings, started_utc, init_duration, agg, process_peak_rss)
    return InferenceStats(
        n_sequences=agg.n_sequences, computed=agg.computed, skipped=agg.skipped
    )


def _compute_missing(
    settings: InferenceSettings, connector: ModelConnector, storage: LogitsStorage
) -> list[_SeqTiming]:
    """Runs the model for every sequence not already in storage, timing each one."""
    timings: list[_SeqTiming] = []
    with tqdm(
        _distinct_sequences(settings.sequences), unit="seq", desc="logits"
    ) as bar:
        for named in bar:
            bar.set_postfix_str(f"{named.name} L={len(named.sequence)}")
            if storage.has(
                model=settings.model,
                sequence=named.sequence,
                start=named.start,
                stop=named.stop,
            ):
                timings.append(
                    _SeqTiming(
                        key=logits_key(named.sequence, named.start, named.stop),
                        length=named.stop - named.start + 1,
                        duration_s=0.0,
                        computed=False,
                        gpu_peak_bytes=None,
                    )
                )
                bar.set_postfix_str(f"{named.name} stored")
                continue

            start = perf_counter()
            result = connector.masked_sequence_logits(
                named.sequence, named.start, named.stop
            )
            duration = perf_counter() - start
            storage.save(
                result,
                model=settings.model,
                label=named.name,
                metadata=named.metadata,
            )
            timings.append(
                _SeqTiming(
                    key=logits_key(named.sequence, named.start, named.stop),
                    length=len(result.residues),
                    duration_s=duration,
                    computed=True,
                    gpu_peak_bytes=connector.peak_memory_bytes(),
                )
            )
            bar.set_postfix_str(f"{named.name} {duration:.2f}s")
    return timings


@dataclass(frozen=True)
class _RunAgg:
    """Aggregate of one run's per-sequence timings, shared by the two report sinks."""

    n_sequences: int
    total_residues: int
    inference_duration: float
    mean_ms_per_residue: float
    computed: int
    skipped: int
    gpu_peak_bytes: int | None


def _aggregate(timings: list[_SeqTiming]) -> _RunAgg:
    """Reduces per-sequence timings to the run-level metrics both report sinks need.

    Residues, duration and GPU peak count only sequences that actually ran:
    dividing a full residue count by near-zero skipped durations would make
    ``mean_ms_per_residue`` meaningless on a mostly-stored run, and
    ``peak_memory_bytes`` still reports the previous call's value when nothing
    ran, which would otherwise leak a stale peak into the aggregate.
    """
    computed_timings = [timing for timing in timings if timing.computed]
    total_residues = sum(timing.length for timing in computed_timings)
    inference_duration = sum(timing.duration_s for timing in computed_timings)
    mean_ms_per_residue = (
        inference_duration / total_residues * 1000 if total_residues else 0.0
    )
    gpu_peaks = [
        timing.gpu_peak_bytes
        for timing in computed_timings
        if timing.gpu_peak_bytes is not None
    ]
    return _RunAgg(
        n_sequences=len(timings),
        total_residues=total_residues,
        inference_duration=inference_duration,
        mean_ms_per_residue=mean_ms_per_residue,
        computed=len(computed_timings),
        skipped=len(timings) - len(computed_timings),
        gpu_peak_bytes=max(gpu_peaks) if gpu_peaks else None,
    )


def _format_bytes(num_bytes: int | None) -> str:
    """Renders a byte count as a human-readable string, or ``n/a`` for ``None``."""
    if num_bytes is None:
        return "n/a"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.0f} {unit}"
        num_bytes = int(num_bytes / 1024)
    return f"{num_bytes:.0f} PB"


def _print_perf_summary(
    settings: InferenceSettings,
    init_duration: float,
    agg: _RunAgg,
    process_peak_rss: int,
) -> None:
    """Prints the end-of-run performance summary to stdout.

    Called after the progress bar closes. Reports sequence/residue counts, init
    and inference durations, mean per-residue latency, computed/skipped counts,
    and the peak process RSS plus the max GPU peak across sequences.
    """
    print(
        f"\n=== performance ({settings.backend}/{settings.model}) ===\n"
        f"sequences: {agg.n_sequences} ({agg.total_residues} residues computed)\n"
        f"init: {init_duration:.3f}s, inference: {agg.inference_duration:.3f}s, "
        f"mean: {agg.mean_ms_per_residue:.3f} ms/residue\n"
        f"storage: {agg.computed} computed / {agg.skipped} already stored\n"
        f"process RSS peak: {_format_bytes(process_peak_rss)}\n"
        f"GPU peak: {_format_bytes(agg.gpu_peak_bytes)}\n"
        f"storage: {settings.storage.describe()}\n"
        f"perf report: {settings.perf_report}"
    )


def _append_perf_csv(
    settings: InferenceSettings,
    started_utc: str,
    init_duration: float,
    agg: _RunAgg,
    process_peak_rss: int,
) -> None:
    """Appends one aggregate row to the longitudinal perf-report CSV.

    One row per :func:`run_inference` invocation; the header is written only
    when the file is first created. GPU peak is the max across computed
    sequences (empty when no backend observed memory).
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
        str(agg.computed),
        str(agg.skipped),
        str(process_peak_rss),
        gpu_peak,
        settings.storage.describe(),
    ]
    is_new = not settings.perf_report.exists()
    settings.perf_report.parent.mkdir(parents=True, exist_ok=True)
    with settings.perf_report.open("a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if is_new:
            writer.writerow(PERF_COLUMNS)
        writer.writerow(row)
