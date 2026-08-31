"""Compute masked ESMC logits for protein sequences and persist them."""

import argparse
import sys
from pathlib import Path
from typing import cast

from esmlab.connectors import ALL_BACKEND_PARAMS, BACKEND_PARAMS, BACKENDS
from esmlab.connectors.base import (
    CANONICAL_SEQUENCE_MODELS,
    load_env,
    resolve_params,
)
from esmlab.inference import InferenceSettings, run_inference
from esmlab.seqio import parse_sequences

DEFAULT_STORAGE = Path("data/logits")


def _build_parser() -> argparse.ArgumentParser:
    """Builds the argparse parser, including the per-backend ParamSpec flags.

    Core flags are declared directly; per-backend flags are generated from the
    co-located ``ParamSpec`` tuples so adding a parameter only touches its own
    module. Defaults of ``None`` let :func:`resolve_params` distinguish "not
    given" from "given".
    """
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
        choices=CANONICAL_SEQUENCE_MODELS,
        default="esmc-600m",
        help="ESMC model size, supported by every backend (default: esmc-600m)",
    )
    parser.add_argument(
        "--storage",
        type=Path,
        default=DEFAULT_STORAGE,
        dest="storage_root",
        help=(
            f"Directory holding computed logits (default: {DEFAULT_STORAGE}). "
            "Use a separate one for stub runs; the root is the only scope."
        ),
    )
    parser.add_argument(
        "--perf-report",
        type=Path,
        default=Path("data/performance.csv"),
        help="Longitudinal perf CSV appended to per run (default: data/performance.csv)",
    )
    # Backend parameters are built from co-located ParamSpec tuples.
    # default=None lets resolve_params tell "not given" from "given".
    for spec in ALL_BACKEND_PARAMS:
        parser.add_argument(
            spec.flag,
            dest=spec.dest,
            default=None,
            type=spec.type,
            choices=spec.choices,
            help=spec.help,
        )
    return parser


def _compute(args: argparse.Namespace) -> int:
    """Logic for one run: resolve backend params, build settings, compute logits.

    Loads ``.env``, resolves backend parameters via :func:`resolve_params`
    (which validates that only relevant flags were given), parses the input
    sequences, assembles an :class:`InferenceSettings`, and calls
    :func:`run_inference`. Returns the process exit code.
    """
    load_env()
    backend_cli: dict[str, str | int | Path | None] = {
        "device": args.device,
        "batch_size": args.batch_size,
        "biohub_api_key": args.biohub_api_key,
        "modal_token_id": args.modal_token_id,
        "modal_token_secret": args.modal_token_secret,
        "modal_gpu": args.modal_gpu,
    }
    backend_resolved = resolve_params(
        BACKEND_PARAMS[args.backend], backend_cli, label=f"backend {args.backend!r}"
    )

    settings = InferenceSettings(
        backend=args.backend,
        model=args.model,
        device=cast(str, backend_resolved.get("device", "")),
        batch_size=cast(int, backend_resolved.get("batch_size", 0)),
        biohub_api_key=cast(str, backend_resolved.get("biohub_api_key", "")),
        modal_token_id=cast(str, backend_resolved.get("modal_token_id", "")),
        modal_token_secret=cast(str, backend_resolved.get("modal_token_secret", "")),
        modal_gpu=cast(str, backend_resolved.get("modal_gpu", "")),
        sequences=parse_sequences(args.sequences, args.fasta),
        storage_root=args.storage_root,
        perf_report=args.perf_report,
    )
    stats = run_inference(settings)
    print(
        f"\n{stats.computed} computed, {stats.skipped} already stored "
        f"({stats.n_sequences} sequence(s)); build reports with:\n"
        f"  mutation_report.py --storage {settings.storage_root} "
        f"--model {settings.model}"
    )
    return 0


def main() -> int:
    """Entrypoint: parse CLI args and delegate to :func:`_compute`.

    Per the repo's script convention this only parses parameters and forwards
    to the logic function; ``ValueError`` from validation is surfaced through
    argparse as a CLI error. Returns the exit code.
    """
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return _compute(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
