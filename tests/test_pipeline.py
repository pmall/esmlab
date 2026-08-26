from pathlib import Path

from esmlab.pipeline import InferSettings, ReportSettings, run_inference, run_report
from esmlab.seqio import NamedSequence

SEQUENCES = [NamedSequence("tiny", "ACDEFGHIKLMNPQRSTVWY")]


def _infer_settings(out_dir: Path) -> InferSettings:
    return InferSettings(
        backend="stub",
        model="esmc-600m",
        device="cpu",
        batch_size=8,
        api_key="",
        sequences=SEQUENCES,
        out_dir=out_dir,
    )


def test_stub_pipeline_produces_all_artifacts(tmp_path: Path) -> None:
    run_paths = run_inference(_infer_settings(tmp_path))
    assert [path.name for path in run_paths] == ["tiny"]

    artifacts = run_report(ReportSettings(run_dirs=run_paths, threshold=0.8, top_k=10))
    artifact_names = {artifact.name for artifact in artifacts}
    assert artifact_names == {
        "entropy.png",
        "deleterious_fraction.png",
        "llr_heatmap.png",
        "summary.csv",
    }

    summary_lines = (run_paths[0] / "summary.csv").read_text().splitlines()
    assert len(summary_lines) == 21  # header + 20 positions
    assert summary_lines[0] == (
        "position,wt_aa,entropy_bits,fraction_deleterious,tolerant,top_alt,top_alt_llr"
    )


def test_report_is_rerunnable_without_backend_change(tmp_path: Path) -> None:
    run_paths = run_inference(_infer_settings(tmp_path))

    def tolerant_column(threshold: float) -> list[str]:
        run_report(ReportSettings(run_dirs=run_paths, threshold=threshold, top_k=5))
        csv_path = next(
            path for path in run_paths[0].iterdir() if path.name == "summary.csv"
        )
        return [line.split(",")[4] for line in csv_path.read_text().splitlines()[1:]]

    strict = tolerant_column(0.5)
    loose = tolerant_column(0.9)
    assert strict != loose
    assert strict.count("True") < loose.count("True")
