"""Compute masked ESMC logits for protein sequences and persist them.

The compute phase of the peptides topic: this runs the model, and
``peptides_report.py`` renders what it stored. It computes masked logits and
persists them to ``data/peptides.sqlite``, this topic's own store; the module
it drives, :mod:`esmlab.inference`, is topic-agnostic and
``sequences_logits.py`` drives it too, into a store of its own.

Sequences come from positional arguments and/or ``--fasta`` files (repeatable).
A FASTA header is ``>label|start|stop`` with an optional ``|{...}`` metadata
object: the coordinates name the sub-sequence to mask and are 1-based and
inclusive, and only those residues are scored, though the whole sequence is
what the model reads. See :mod:`esmlab.seqio`. A request already in storage is
skipped rather than recomputed, so re-running over a grown FASTA only computes
the new records, and a record listed twice is one unit of work.

Backend flags (``--backend`` and its credentials) and storage flags
(``--storage`` and its connection parameters) are generated from the
``ParamSpec`` tuples co-located with each backend and with
:mod:`esmlab.storage`, so a flag belonging to something you did not select is
an error rather than being silently ignored. Most fall back to an environment
variable read from ``.env``; see ``.env.example``, and ``--help`` for the
current list. The run ends by printing the ``peptides_report.py`` command line
that reopens the store it just wrote.

Each run also appends one aggregate row to ``--perf-report``, which is
longitudinal across runs rather than per-run output.
"""

import argparse
import sys
from pathlib import Path
from typing import cast

from esmlab.connectors import ALL_BACKEND_PARAMS, BACKEND_PARAMS, BACKENDS
from esmlab.connectors.base import CANONICAL_SEQUENCE_MODELS
from esmlab.inference import InferenceSettings, run_inference
from esmlab.params import load_env, resolve_params
from esmlab.seqio import parse_sequences
from esmlab.storage import add_storage_arguments, storage_settings

SUMMARY = __doc__.partition("\n\n")[0] if __doc__ else ""

# The store this topic writes and reads. One database per topic, so a report
# covers this topic's runs and nothing else.
SQLITE_PATH = Path("data/peptides.sqlite")


def _build_parser() -> argparse.ArgumentParser:
    """Builds the argparse parser, including the per-backend and storage flags.

    Core flags are declared directly; per-backend flags are generated from the
    co-located ``ParamSpec`` tuples so adding a parameter only touches its own
    module, and the storage flags come from :func:`add_storage_arguments` so
    both stages accept the same vocabulary. Defaults of ``None`` let
    :func:`resolve_params` distinguish "not given" from "given".
    """
    # argparse takes the summary line only: the rest of the module docstring
    # is for someone reading the file, and would swamp --help.
    parser = argparse.ArgumentParser(description=SUMMARY)
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
    add_storage_arguments(parser, sqlite_default=SQLITE_PATH)
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
    """Logic for one run: resolve backend and storage params, then compute logits.

    Loads ``.env``, resolves backend and storage parameters via
    :func:`resolve_params` (which validates that only relevant flags were
    given), parses the input sequences, assembles an
    :class:`InferenceSettings`, and calls :func:`run_inference`. Returns the
    process exit code.
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
        storage=storage_settings(args, sqlite_default=SQLITE_PATH),
        perf_report=args.perf_report,
    )
    stats = run_inference(settings)
    print(
        f"\n{stats.computed} computed, {stats.skipped} already stored "
        f"({stats.n_sequences} sequence(s)); build reports with:\n"
        f"  peptides_report.py {settings.storage.flags()} "
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
