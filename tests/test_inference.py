"""Covers the compute stage: store-first, skip-if-present, dedup, perf CSV."""

import csv
from pathlib import Path

from esmlab.inference import PERF_COLUMNS, InferenceSettings, run_inference
from esmlab.seqio import NamedSequence, parse_sequences
from esmlab.storage import open_storage
from tests.fixtures import sqlite_settings

SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"
OTHER_SEQUENCE = "MKTAYIAKQRQISFVK"


def _settings(
    tmp_path: Path, sequences: list[NamedSequence] | None = None
) -> InferenceSettings:
    """Builds an :class:`InferenceSettings` wired to the stub backend.

    The stub needs no credentials, so the backend fields are blanked. The
    SQLite file and the perf CSV both live under ``tmp_path`` so tests never
    touch the real ``data/`` tree.
    """
    return InferenceSettings(
        backend="stub",
        model="esmc-600m",
        device="",
        batch_size=0,
        biohub_api_key="",
        modal_token_id="",
        modal_token_secret="",
        modal_gpu="",
        sequences=sequences or [NamedSequence("tiny", SEQUENCE)],
        storage=sqlite_settings(tmp_path / "logits.sqlite3"),
        perf_report=tmp_path / "performance.csv",
    )


def _perf_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def test_run_stores_one_entry_per_sequence(tmp_path: Path) -> None:
    """A first run computes every sequence and persists it under its label."""
    settings = _settings(tmp_path)

    stats = run_inference(settings)

    assert (stats.n_sequences, stats.computed, stats.skipped) == (1, 1, 0)
    storage = open_storage(settings.storage)
    assert storage.has(model="esmc-600m", sequence=SEQUENCE) is True
    assert storage.load(model="esmc-600m", sequence=SEQUENCE).label == "tiny"


def test_second_identical_run_skips_everything(tmp_path: Path) -> None:
    """Re-running computes nothing and adds no entries: the point of the split."""
    settings = _settings(tmp_path)
    run_inference(settings)

    stats = run_inference(settings)

    assert (stats.n_sequences, stats.computed, stats.skipped) == (1, 0, 1)
    assert len(list(open_storage(settings.storage).entries(model="esmc-600m"))) == 1


def test_duplicate_sequences_compute_once(tmp_path: Path) -> None:
    """A sequence listed twice is one unit of work, not one computed and one skipped."""
    settings = _settings(
        tmp_path,
        [NamedSequence("first", SEQUENCE), NamedSequence("second", SEQUENCE)],
    )

    stats = run_inference(settings)

    assert (stats.n_sequences, stats.computed, stats.skipped) == (1, 1, 0)
    # The first-seen label wins, since dedup keeps the first occurrence.
    storage = open_storage(settings.storage)
    assert storage.load(model="esmc-600m", sequence=SEQUENCE).label == "first"


def test_distinct_sequences_sharing_a_label_are_both_stored(tmp_path: Path) -> None:
    """Labels are display-only, so a shared label never collapses two sequences."""
    settings = _settings(
        tmp_path,
        [NamedSequence("dup", SEQUENCE), NamedSequence("dup", OTHER_SEQUENCE)],
    )

    stats = run_inference(settings)

    assert (stats.n_sequences, stats.computed) == (2, 2)
    storage = open_storage(settings.storage)
    assert storage.has(model="esmc-600m", sequence=SEQUENCE) is True
    assert storage.has(model="esmc-600m", sequence=OTHER_SEQUENCE) is True


def test_perf_csv_gets_a_header_and_one_row_per_run(tmp_path: Path) -> None:
    """The perf CSV is longitudinal: header once, one aggregate row per run."""
    settings = _settings(tmp_path)

    run_inference(settings)
    rows = _perf_rows(settings.perf_report)
    assert list(rows[0]) == list(PERF_COLUMNS)
    assert rows[0]["backend"] == "stub"
    assert rows[0]["model"] == "esmc-600m"
    assert rows[0]["computed"] == "1"
    assert rows[0]["skipped"] == "0"
    assert rows[0]["total_residues"] == str(len(SEQUENCE))
    # The stub cannot observe GPU memory, so the column stays empty.
    assert rows[0]["gpu_peak_bytes"] == ""
    assert rows[0]["storage"] == settings.storage.describe()

    run_inference(settings)
    rows = _perf_rows(settings.perf_report)
    assert len(rows) == 2
    assert rows[1]["computed"] == "0"
    assert rows[1]["skipped"] == "1"


def test_skipped_run_reports_no_residues_or_duration(tmp_path: Path) -> None:
    """Only computed sequences feed the timing math, so a full skip reports zero."""
    settings = _settings(tmp_path)
    run_inference(settings)
    run_inference(settings)

    skipped_row = _perf_rows(settings.perf_report)[1]
    assert skipped_row["total_residues"] == "0"
    assert float(skipped_row["mean_ms_per_residue"]) == 0.0


def test_perf_summary_is_printed(tmp_path: Path, capsys) -> None:
    """The end-of-run summary block still reaches stdout."""
    run_inference(_settings(tmp_path))

    out = capsys.readouterr().out
    assert "=== performance (stub/esmc-600m) ===" in out
    assert "1 computed / 0 already stored" in out
    assert "GPU peak: n/a" in out


def test_header_metadata_reaches_storage(tmp_path: Path) -> None:
    """A header's JSON survives parsing, inference and persistence unchanged.

    The end of the `>label|{...}` path: the label lands in its own column and
    the object in the JSON one, so both are queryable.
    """
    fasta = tmp_path / "input.fa"
    fasta.write_text(
        f'>stat1|{{"source": "UniProt:P12345", "targets": ["P11111"]}}\n{SEQUENCE}\n'
    )
    settings = _settings(tmp_path, parse_sequences([], [fasta]))

    run_inference(settings)

    stored = open_storage(settings.storage).load(model="esmc-600m", sequence=SEQUENCE)
    assert stored.label == "stat1"
    assert stored.metadata == {"source": "UniProt:P12345", "targets": ["P11111"]}
