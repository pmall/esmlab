"""Build mutation-analysis reports from stored scoring runs."""

import argparse
import sys
from pathlib import Path

from esmlab.pipeline import ReportSettings, run_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Run directories produced by mutation_scoring.py",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="A position counts as mutation-tolerant below this deleterious fraction",
    )
    parser.add_argument(
        "--top", type=int, default=20, help="Number of substitutions to rank"
    )
    return parser


def _report_runs(args: argparse.Namespace) -> int:
    missing = [path for path in args.run_dirs if not path.is_dir()]
    if missing:
        print(
            f"Not a run directory: {', '.join(str(path) for path in missing)}",
            file=sys.stderr,
        )
        return 2
    artifacts = run_report(
        ReportSettings(run_dirs=args.run_dirs, threshold=args.threshold, top_k=args.top)
    )
    print(f"\nWrote {len(artifacts)} artifact(s):")
    for artifact in artifacts:
        print(f"  {artifact}")
    return 0


def main() -> int:
    return _report_runs(_build_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
