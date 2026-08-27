from pathlib import Path

import pytest

from esmlab.pipeline import AnalysisSettings, run_analysis
from esmlab.seqio import NamedSequence

SEQUENCES = [NamedSequence("tiny", "ACDEFGHIKLMNPQRSTVWY")]


def _settings(
    out_dir: Path,
    *,
    threshold: float = 0.8,
    cache: str = "null",
    cache_root: Path | None = None,
) -> AnalysisSettings:
    """Builds an :class:`AnalysisSettings` wired to the stub backend.

    The stub needs no real credentials, so the backend fields are blanked;
    callers override ``threshold``/``cache`` to exercise specific paths. The
    perf-report CSV defaults to ``<out_dir>/performance.csv`` so tests append
    to a tmp_path-local file rather than the real ``data/`` tree.
    """
    return AnalysisSettings(
        backend="stub",
        model="esmc-600m",
        device="",
        batch_size=0,
        forge_api_key="",
        modal_token_id="",
        modal_token_secret="",
        sequences=SEQUENCES,
        out_dir=out_dir,
        threshold=threshold,
        top_k=10,
        cache=cache,
        cache_root=cache_root,
        perf_report=out_dir / "performance.csv",
    )


def test_analysis_produces_all_artifacts(tmp_path: Path) -> None:
    """One run writes the three plots plus the summary CSV, with the expected header."""
    artifacts = run_analysis(_settings(tmp_path))
    assert {path.name for path in artifacts} == {
        "entropy.png",
        "deleterious_fraction.png",
        "llr_heatmap.png",
        "summary.csv",
    }

    summary_lines = (tmp_path / "tiny" / "summary.csv").read_text().splitlines()
    assert len(summary_lines) == 21  # header + 20 positions
    assert summary_lines[0] == (
        "position,wt_aa,entropy_bits,fraction_deleterious,tolerant,top_alt,top_alt_llr"
    )


def test_analysis_is_rerunnable_with_new_threshold(tmp_path: Path) -> None:
    """Re-running with a looser threshold marks more positions tolerant."""

    def tolerant_column(threshold: float) -> list[str]:
        """Runs the analysis at ``threshold`` and returns the tolerant column."""
        run_analysis(_settings(tmp_path, threshold=threshold))
        return [
            line.split(",")[4]
            for line in (tmp_path / "tiny" / "summary.csv").read_text().splitlines()[1:]
        ]

    strict = tolerant_column(0.5)
    loose = tolerant_column(0.9)
    assert strict != loose
    assert strict.count("True") < loose.count("True")


def test_analysis_with_file_cache_stores_logits(tmp_path: Path) -> None:
    """With the file cache, one run writes a single logits+meta cache entry."""
    cache_dir = tmp_path / "cache"
    run_analysis(_settings(tmp_path, cache="file", cache_root=cache_dir))

    entries = list(cache_dir.iterdir())
    assert len(entries) == 1
    entry = entries[0]
    assert (entry / "logits.npz").is_file()
    assert (entry / "meta.json").is_file()
    assert (tmp_path / "tiny" / "summary.csv").is_file()


def test_perf_report_appends_one_row_per_run(tmp_path: Path) -> None:
    """The perf CSV is created with a header, then appends one row per run."""
    run_analysis(_settings(tmp_path))
    run_analysis(_settings(tmp_path))

    lines = (tmp_path / "performance.csv").read_text().splitlines()
    assert len(lines) == 3  # header + two runs
    header = lines[0].split(",")
    assert header == [
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
    ]
    for row in (lines[1], lines[2]):
        cells = dict(zip(header, row.split(",")))
        assert cells["backend"] == "stub"
        assert cells["model"] == "esmc-600m"
        assert cells["n_sequences"] == "1"
        assert cells["cache_misses"] == "1"
        assert cells["gpu_peak_bytes"] == ""  # stub cannot observe GPU memory


def test_perf_report_stdout_summary_prints_after_run(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """A performance summary block is printed to stdout after the run completes."""
    run_analysis(_settings(tmp_path))
    out, _ = capfd.readouterr()
    assert "=== performance (stub/esmc-600m) ===" in out
    assert "process RSS peak:" in out
    assert "GPU peak: n/a" in out
