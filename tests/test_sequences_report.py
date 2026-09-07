"""Covers the whole-sequence report stage: a page per entry, the ranking index, free re-runs."""

import json
import re
from pathlib import Path

import pytest

from esmlab.inference import InferenceSettings, run_inference
from esmlab.seqio import NamedSequence
from esmlab.sequences_report import ReportSettings, run_report
from esmlab.storage import StorageSettings, logits_key
from tests.fixtures import named, sqlite_settings

SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"
OTHER_SEQUENCE = "MKTAYIAKQRQISFVK"


def _populate(
    tmp_path: Path,
    sequences: list[NamedSequence],
    *,
    model: str = "esmc-600m",
) -> StorageSettings:
    """Runs the stub compute stage so the report stage has entries to read."""
    storage = sqlite_settings(tmp_path / "logits.sqlite3")
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
            storage=storage,
            perf_report=tmp_path / "performance.csv",
        )
    )
    return storage


def _settings(
    tmp_path: Path, *, window: int = 9, model: str | None = "esmc-600m"
) -> ReportSettings:
    return ReportSettings(
        model=model,
        storage=sqlite_settings(tmp_path / "logits.sqlite3"),
        out_dir=tmp_path / "reports",
        window=window,
    )


def _payload(page: Path) -> dict:
    """Reads back the JSON island a rendered page carries."""
    island = re.search(
        r'<script type="application/json" id="payload">(.*?)</script>',
        page.read_text(),
        re.DOTALL,
    )
    assert island is not None
    return json.loads(island.group(1))


def test_a_run_writes_one_page_per_entry_and_an_index(tmp_path: Path) -> None:
    """The page is the whole report: no sidecar files accompany it."""
    _populate(tmp_path, [named("tiny", SEQUENCE), named("other", OTHER_SEQUENCE)])
    settings = _settings(tmp_path)

    run_report(settings)

    assert {
        path.name for path in (settings.out_dir / "sequences" / "esmc-600m").iterdir()
    } == {
        f"{logits_key(SEQUENCE, 1, len(SEQUENCE))}.html",
        f"{logits_key(OTHER_SEQUENCE, 1, len(OTHER_SEQUENCE))}.html",
        "index.html",
    }


def test_a_whole_sequence_entry_is_scored_over_every_residue(tmp_path: Path) -> None:
    """A record with no region of its own reports one entropy per residue."""
    _populate(tmp_path, [named("tiny", SEQUENCE)])
    settings = _settings(tmp_path)

    run_report(settings)

    payload = _payload(
        settings.out_dir
        / "sequences"
        / "esmc-600m"
        / f"{logits_key(SEQUENCE, 1, len(SEQUENCE))}.html"
    )
    assert payload["sequence"] == SEQUENCE
    assert (payload["start"], payload["stop"]) == (1, len(SEQUENCE))
    assert len(payload["entropy"]) == len(SEQUENCE)


def test_rerun_with_a_new_window_touches_no_storage(tmp_path: Path) -> None:
    """Smoothing is presentation: changing it re-renders and computes nothing."""
    _populate(tmp_path, [named("tiny", SEQUENCE)])
    database = tmp_path / "logits.sqlite3"
    before = database.stat().st_mtime_ns

    run_report(_settings(tmp_path, window=1))
    run_report(_settings(tmp_path, window=15))

    assert database.stat().st_mtime_ns == before
    page = _settings(tmp_path).out_dir / "sequences" / "esmc-600m"
    payload = _payload(page / f"{logits_key(SEQUENCE, 1, len(SEQUENCE))}.html")
    assert payload["smoothed"] != payload["entropy"]


def test_index_ranks_every_entry_of_the_model(tmp_path: Path) -> None:
    """The listing covers the run and orders it by mean entropy."""
    _populate(tmp_path, [named("tiny", SEQUENCE), named("other", OTHER_SEQUENCE)])
    settings = _settings(tmp_path)

    run_report(settings)

    payload = _payload(settings.out_dir / "sequences" / "esmc-600m" / "index.html")
    means = [row["mean"] for row in payload["entries"]]
    assert len(means) == 2
    assert means == sorted(means)


def test_no_model_reports_every_model_in_one_run(tmp_path: Path) -> None:
    """Reporting costs no model time, so the default is every model stored."""
    _populate(tmp_path, [named("tiny", SEQUENCE)], model="esmc-300m")
    _populate(tmp_path, [named("tiny", SEQUENCE)], model="esmc-600m")
    settings = _settings(tmp_path, model=None)

    run_report(settings)

    assert (settings.out_dir / "sequences" / "esmc-300m" / "index.html").is_file()
    assert (settings.out_dir / "sequences" / "esmc-600m" / "index.html").is_file()


def test_reporting_an_empty_storage_is_an_error(tmp_path: Path) -> None:
    """A selection matching nothing must not leave an empty directory behind."""
    settings = _settings(tmp_path)

    with pytest.raises(ValueError, match="no entries at all"):
        run_report(settings)

    assert not settings.out_dir.exists()
