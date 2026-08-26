"""Score protein sequences with a masked ESMC model and store reusable logits runs."""

import argparse
import os
import sys
from pathlib import Path

from esmlab.connectors import BACKENDS
from esmlab.connectors.base import CANONICAL_MODELS
from esmlab.pipeline import InferSettings, run_inference
from esmlab.seqio import parse_sequences


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequences", nargs="*", help="Raw amino acid sequences")
    parser.add_argument(
        "--fasta",
        action="append",
        type=Path,
        default=[],
        help="FASTA file with input sequences (repeatable)",
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="stub",
        help="Inference backend (default: stub, no model required)",
    )
    parser.add_argument(
        "--model",
        choices=CANONICAL_MODELS,
        default="esmc-600m",
        help="Model size, supported by every backend (default: esmc-600m)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device for the local backend (default: auto-detect)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Masked variants per forward pass on the local backend",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ESM_API_KEY", ""),
        help="Forge API token (defaults to $ESM_API_KEY)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs"), help="Directory for run folders"
    )
    return parser


def _score_sequences(args: argparse.Namespace) -> int:
    run_paths = run_inference(
        InferSettings(
            backend=args.backend,
            model=args.model,
            device=args.device,
            batch_size=args.batch_size,
            api_key=args.api_key,
            sequences=parse_sequences(args.sequences, args.fasta),
            out_dir=args.out,
        )
    )
    print(f"Scored {len(run_paths)} sequence(s); build reports with:")
    print(f"  mutation_report.py {' '.join(str(path) for path in run_paths)}")
    return 0


def main() -> int:
    return _score_sequences(_build_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
