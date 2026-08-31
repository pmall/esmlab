"""Score protein sequences with a masked ESMC model and build mutation reports."""

import argparse
import sys
from pathlib import Path
from typing import cast

from esmlab.cache import ALL_CACHE_PARAMS, CACHE_PARAMS, CACHES
from esmlab.connectors import ALL_BACKEND_PARAMS, BACKEND_PARAMS, BACKENDS
from esmlab.connectors.base import CANONICAL_MODELS, load_env, resolve_params
from esmlab.pipeline import AnalysisSettings, run_analysis
from esmlab.seqio import parse_sequences


def _build_parser() -> argparse.ArgumentParser:
    """Builds the argparse parser, including backend/cache ParamSpec flags.

    Core flags (sequences, --fasta, --backend, --model, --threshold, --top,
    --out, --cache) are declared directly; per-backend and per-cache flags
    are generated from the co-located ``ParamSpec`` tuples so adding a
    parameter only touches its own module. Defaults of ``None`` let
    :func:`resolve_params` distinguish "not given" from "given".
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
        choices=CANONICAL_MODELS,
        default="esmc-600m",
        help="Model size, supported by every backend (default: esmc-600m)",
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
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/reports"),
        help="Directory for per-sequence report folders (default: data/reports)",
    )
    parser.add_argument(
        "--cache",
        choices=CACHES,
        default="null",
        help="Logits cache store (default: null, no caching)",
    )
    parser.add_argument(
        "--perf-report",
        type=Path,
        default=Path("data/performance.csv"),
        help="Longitudinal perf CSV appended to per run (default: data/performance.csv)",
    )
    # Backend and cache parameters are built from co-located ParamSpec tuples.
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
    for spec in ALL_CACHE_PARAMS:
        parser.add_argument(
            spec.flag,
            dest=spec.dest,
            default=None,
            type=spec.type,
            choices=spec.choices,
            help=spec.help,
        )
    return parser


def _analyze(args: argparse.Namespace) -> int:
    """Logic for one run: resolve params, build settings, run the analysis.

    Loads ``.env``, resolves backend and cache parameters via
    :func:`resolve_params` (which validates that only relevant flags were
    given), parses the input sequences into :class:`NamedSequence` objects,
    assembles an :class:`AnalysisSettings`, calls :func:`run_analysis`, and
    prints the list of written artifacts. Returns the process exit code.
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
    cache_cli: dict[str, str | int | Path | None] = {"cache_root": args.cache_root}

    backend_resolved = resolve_params(
        BACKEND_PARAMS[args.backend], backend_cli, label=f"backend {args.backend!r}"
    )
    cache_resolved = resolve_params(
        CACHE_PARAMS[args.cache], cache_cli, label=f"cache {args.cache!r}"
    )

    settings = AnalysisSettings(
        backend=args.backend,
        model=args.model,
        device=cast(str, backend_resolved.get("device", "")),
        batch_size=cast(int, backend_resolved.get("batch_size", 0)),
        biohub_api_key=cast(str, backend_resolved.get("biohub_api_key", "")),
        modal_token_id=cast(str, backend_resolved.get("modal_token_id", "")),
        modal_token_secret=cast(str, backend_resolved.get("modal_token_secret", "")),
        modal_gpu=cast(str, backend_resolved.get("modal_gpu", "")),
        sequences=parse_sequences(args.sequences, args.fasta),
        out_dir=args.out,
        threshold=args.threshold,
        top_k=args.top,
        cache=args.cache,
        cache_root=cast(Path | None, cache_resolved.get("cache_root")),
        perf_report=args.perf_report,
    )
    artifacts = run_analysis(settings)
    print(f"\nWrote {len(artifacts)} artifact(s):")
    for artifact in artifacts:
        print(f"  {artifact}")
    return 0


def main() -> int:
    """Entrypoint: parse CLI args and delegate to :func:`_analyze`.

    Per the repo's script convention this only parses parameters and forwards
    to the logic function; ``ValueError`` from validation is surfaced through
    argparse as a CLI error. Returns the exit code.
    """
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return _analyze(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
