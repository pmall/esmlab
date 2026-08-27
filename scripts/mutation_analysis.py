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
    load_env()
    backend_cli: dict[str, str | int | Path | None] = {
        "device": args.device,
        "batch_size": args.batch_size,
        "forge_api_key": args.forge_api_key,
        "modal_token_id": args.modal_token_id,
        "modal_token_secret": args.modal_token_secret,
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
        forge_api_key=cast(str, backend_resolved.get("forge_api_key", "")),
        modal_token_id=cast(str, backend_resolved.get("modal_token_id", "")),
        modal_token_secret=cast(str, backend_resolved.get("modal_token_secret", "")),
        sequences=parse_sequences(args.sequences, args.fasta),
        out_dir=args.out,
        threshold=args.threshold,
        top_k=args.top,
        cache=args.cache,
        cache_root=cast(Path | None, cache_resolved.get("cache_root")),
    )
    artifacts = run_analysis(settings)
    print(f"\nWrote {len(artifacts)} artifact(s):")
    for artifact in artifacts:
        print(f"  {artifact}")
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return _analyze(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
