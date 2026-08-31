"""Build mutation reports from logits already persisted by mutation_logits.py."""

import argparse
import sys
from pathlib import Path

from esmlab.connectors.base import CANONICAL_SEQUENCE_MODELS
from esmlab.mutation_report import ReportSettings, run_report

DEFAULT_STORAGE = Path("data/logits")


def _build_parser() -> argparse.ArgumentParser:
    """Builds the argparse parser.

    No backend flags and no credentials: this stage reads stored arrays and
    never constructs a connector.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=CANONICAL_SEQUENCE_MODELS,
        default="esmc-600m",
        help="Report on entries stored for this model (default: esmc-600m)",
    )
    parser.add_argument(
        "--storage",
        type=Path,
        default=DEFAULT_STORAGE,
        dest="storage_root",
        help=f"Directory holding computed logits (default: {DEFAULT_STORAGE})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/reports"),
        help="Directory for report artifacts (default: data/reports)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Deleterious-fraction threshold below which a position is tolerant",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        dest="top_k",
        help="How many top substitutions to rank per sequence (default: 20)",
    )
    return parser


def _report(args: argparse.Namespace) -> int:
    """Logic for one run: build settings and render every stored entry.

    Assembles a :class:`ReportSettings`, calls :func:`run_report`, and prints
    the written artifacts. Returns the process exit code.
    """
    settings = ReportSettings(
        model=args.model,
        storage_root=args.storage_root,
        out_dir=args.out,
        threshold=args.threshold,
        top_k=args.top_k,
    )
    artifacts = run_report(settings)
    print(f"\nWrote {len(artifacts)} artifact(s):")
    for artifact in artifacts:
        print(f"  {artifact}")
    return 0


def main() -> int:
    """Entrypoint: parse CLI args and delegate to :func:`_report`.

    Per the repo's script convention this only parses parameters and forwards
    to the logic function; ``ValueError`` from validation is surfaced through
    argparse as a CLI error. Returns the exit code.
    """
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return _report(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
