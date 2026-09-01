"""Build mutation reports from logits already persisted by mutation_logits.py.

A consuming phase over what ``mutation_logits.py`` stored: it reads a storage
and never constructs a connector, so it needs no credentials, no GPU and no
model. Re-running with a different ``--threshold`` or ``--top`` is pure CPU
work over arrays that are already stored; another analysis of the same logits
is a sibling script rather than another model run.

It iterates the storage rather than an input file — the store is the source of
truth for what has been computed — and renders every entry held for ``--model``.
Each entry becomes a self-contained ``<out>/<model>/<key>.html`` page, keyed by
the sequence's storage digest rather than by its label, beside an
``index.html`` linking that model's keys back to labels for humans. The model
is in the path because the key is the sequence's alone, so two models would
otherwise overwrite each other. Open a model's ``index.html`` to read a run; a
page needs a network connection the first time, for the charting library it
loads from a CDN.

``--model`` is optional: the default renders every model the store holds, since
reporting costs no model time. Naming a model with nothing stored is a CLI
error listing the models that do have entries.

Takes the same storage flags as ``mutation_logits.py``.
"""

import argparse
import sys
from pathlib import Path

from esmlab.connectors.base import CANONICAL_SEQUENCE_MODELS
from esmlab.mutation_report import ReportSettings, run_report
from esmlab.params import load_env
from esmlab.storage import add_storage_arguments, storage_settings

SUMMARY = __doc__.partition("\n\n")[0] if __doc__ else ""


def _build_parser() -> argparse.ArgumentParser:
    """Builds the argparse parser.

    No backend flags and no model credentials: this stage reads stored arrays
    and never constructs a connector. The storage flags are the same ones
    ``mutation_logits`` declares, so a run's printed command line is copyable.
    """
    # argparse takes the summary line only: the rest of the module docstring
    # is for someone reading the file, and would swamp --help.
    parser = argparse.ArgumentParser(description=SUMMARY)
    parser.add_argument(
        "--model",
        choices=CANONICAL_SEQUENCE_MODELS,
        default=None,
        help="Report only on entries stored for this model (default: every model)",
    )
    add_storage_arguments(parser)
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
    """Logic for one run: resolve storage params and render every stored entry.

    Loads ``.env`` so the storage env fallbacks resolve, assembles a
    :class:`ReportSettings`, calls :func:`run_report`, and prints the written
    artifacts. Returns the process exit code.
    """
    load_env()
    settings = ReportSettings(
        model=args.model,
        storage=storage_settings(args),
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
