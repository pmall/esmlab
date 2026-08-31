"""Covers the report stage: artifacts per stored entry, manifest, free re-runs."""

import csv
from pathlib import Path

from esmlab.inference import InferenceSettings, run_inference
from esmlab.mutation_report import (
    MANIFEST_COLUMNS,
    SUMMARY_COLUMNS,
    ReportSettings,
    run_report,
)
from esmlab.seqio import NamedSequence
from esmlab.storage import logits_key

SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"
OTHER_SEQUENCE = "MKTAYIAKQRQISFVK"
ARTIFACT_NAMES = {
    "entropy.png",
    "deleterious_fraction.png",
    "llr_heatmap.png",
    "summary.csv",
}


def _populate(
    tmp_path: Path,
    sequences: list[NamedSequence],
    *,
    model: str = "esmc-600m",
) -> Path:
    """Runs the stub compute stage so the report stage has entries to read."""
    run_inference(
        InferenceSettings(
            backend="stub",
            model=model,
            device="",
            batch_size=0,
            biohub_api_key="",
            modal_token_id="",
            modal_token_secret="",
            modal_gpu="",
            sequences=sequences,
            storage_root=tmp_path / "logits",
            perf_report=tmp_path / "performance.csv",
        )
    )
    return tmp_path / "logits"


def _settings(
    tmp_path: Path, *, threshold: float = 0.8, model: str = "esmc-600m"
) -> ReportSettings:
    return ReportSettings(
        model=model,
        storage_root=tmp_path / "logits",
        out_dir=tmp_path / "reports",
        threshold=threshold,
        top_k=10,
    )


def _summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _summary_path(tmp_path: Path, sequence: str = SEQUENCE) -> Path:
    """Where the report stage writes a sequence's summary CSV."""
    return tmp_path / "reports" / "esmc-600m" / logits_key(sequence) / "summary.csv"


def test_report_writes_all_artifacts_per_entry(tmp_path: Path) -> None:
    """Every stored entry gets three plots and a summary CSV under its key."""
    _populate(tmp_path, [NamedSequence("tiny", SEQUENCE)])
    settings = _settings(tmp_path)

    run_report(settings)

    report_dir = settings.out_dir / "esmc-600m" / logits_key(SEQUENCE)
    assert {path.name for path in report_dir.iterdir()} == ARTIFACT_NAMES
    rows = _summary_rows(report_dir / "summary.csv")
    assert list(rows[0]) == list(SUMMARY_COLUMNS)
    assert len(rows) == len(SEQUENCE)


def test_manifest_maps_keys_back_to_labels(tmp_path: Path) -> None:
    """The label survives only as manifest data, never as a path component."""
    _populate(tmp_path, [NamedSequence("tiny", SEQUENCE)])
    settings = _settings(tmp_path)

    run_report(settings)

    with (settings.out_dir / "esmc-600m" / "manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(MANIFEST_COLUMNS)
    assert rows[0]["key"] == logits_key(SEQUENCE)
    assert rows[0]["label"] == "tiny"
    assert rows[0]["length"] == str(len(SEQUENCE))


def test_rerun_with_new_threshold_touches_no_storage(tmp_path: Path) -> None:
    """Re-rendering is pure CPU work: the tolerant column flips, storage is untouched.

    This is the payoff of splitting compute from reporting - the old unified
    pipeline re-ran the model to change one presentation knob.
    """
    storage_root = _populate(tmp_path, [NamedSequence("tiny", SEQUENCE)])
    entry_dir = storage_root / "esmc-600m" / logits_key(SEQUENCE)
    before = {path.name: path.stat().st_mtime_ns for path in entry_dir.iterdir()}

    summary = _summary_path(tmp_path)
    run_report(_settings(tmp_path, threshold=0.0))
    strict = sum(row["tolerant"] == "True" for row in _summary_rows(summary))

    run_report(_settings(tmp_path, threshold=1.0))
    loose = sum(row["tolerant"] == "True" for row in _summary_rows(summary))

    assert loose > strict
    after = {path.name: path.stat().st_mtime_ns for path in entry_dir.iterdir()}
    assert after == before


def test_two_records_sharing_a_label_both_get_reports(tmp_path: Path) -> None:
    """Duplicate FASTA headers are fine now that keys come from the sequence."""
    _populate(
        tmp_path,
        [NamedSequence("dup", SEQUENCE), NamedSequence("dup", OTHER_SEQUENCE)],
    )
    settings = _settings(tmp_path)

    run_report(settings)

    model_dir = settings.out_dir / "esmc-600m"
    assert (model_dir / logits_key(SEQUENCE)).is_dir()
    assert (model_dir / logits_key(OTHER_SEQUENCE)).is_dir()


def test_two_models_report_side_by_side(tmp_path: Path) -> None:
    """The model is in the report path, so one --out holds both without overwriting."""
    _populate(tmp_path, [NamedSequence("tiny", SEQUENCE)], model="esmc-600m")
    _populate(tmp_path, [NamedSequence("tiny", SEQUENCE)], model="esmc-300m")

    run_report(_settings(tmp_path, model="esmc-600m"))
    run_report(_settings(tmp_path, model="esmc-300m"))

    out_dir = tmp_path / "reports"
    key = logits_key(SEQUENCE)
    assert (out_dir / "esmc-600m" / key / "summary.csv").is_file()
    assert (out_dir / "esmc-300m" / key / "summary.csv").is_file()


def test_empty_storage_writes_only_a_manifest(tmp_path: Path) -> None:
    """Reporting a model with no entries is not an error."""
    settings = _settings(tmp_path)

    artifacts = run_report(settings)

    assert [path.name for path in artifacts] == ["manifest.csv"]
