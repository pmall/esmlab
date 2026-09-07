"""Build peptide reports from logits already persisted by peptides_logits.py.

A consuming phase over what ``peptides_logits.py`` stored in
``data/peptides.sqlite``: it reads a storage
and never constructs a connector, so it needs no credentials, no GPU and no
model. Re-running with a different ``--threshold`` or ``--top`` is pure CPU
work over arrays that are already stored; another analysis of the same logits
is a sibling script rather than another model run.

It iterates the storage rather than an input file — the store is the source of
truth for what has been computed — and renders every entry held for the
selected ``--backend`` and ``--model``. Each entry becomes a self-contained
``<out>/peptides/<backend>/<model>/<key>.html`` page, keyed by the sequence's
storage digest rather than by its label, beside an ``index.html`` linking that
run's keys back to labels for humans. ``--out`` is the reports root every topic
shares (default ``data/reports``); the topic, the backend and the model are in
the path below it because the key is the sequence's alone, so nothing else may
share a directory with it. Open a run's ``index.html`` to read it; a page needs
a network connection the first time, for the charting library it loads from a
CDN.

``--backend`` and ``--model`` are both optional: the default renders every
backend and model the store holds, since reporting costs no model time. Naming
a pair with nothing stored is a CLI error listing the pairs that do have
entries.

Takes the same storage flags as ``peptides_logits.py``, and reads by default
the same ``data/peptides.sqlite`` it writes.
"""

import argparse
import sys
from pathlib import Path

from esmlab.connectors import BACKENDS
from esmlab.connectors.base import CANONICAL_SEQUENCE_MODELS
from esmlab.params import load_env
from esmlab.peptides_report import ReportSettings, run_report
from esmlab.storage import add_storage_arguments, storage_settings

SUMMARY = __doc__.partition("\n\n")[0] if __doc__ else ""

# The store this topic writes and reads. One database per topic, so a report
# covers this topic's runs and nothing else.
SQLITE_PATH = Path("data/peptides.sqlite")


def _build_parser() -> argparse.ArgumentParser:
    """Builds the argparse parser.

    No model credentials, and ``--backend`` here selects stored entries rather
    than a connector: this stage reads stored arrays and never runs a model.
    The storage flags are the same ones ``peptides_logits`` declares, so a
    run's printed command line is copyable.
    """
    # argparse takes the summary line only: the rest of the module docstring
    # is for someone reading the file, and would swamp --help.
    parser = argparse.ArgumentParser(description=SUMMARY)
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default=None,
        help="Report only on entries this backend computed (default: every backend)",
    )
    parser.add_argument(
        "--model",
        choices=CANONICAL_SEQUENCE_MODELS,
        default=None,
        help="Report only on entries stored for this model (default: every model)",
    )
    add_storage_arguments(parser, sqlite_default=SQLITE_PATH)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/reports"),
        help="Root for report artifacts, one directory per topic (default: data/reports)",
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
        backend=args.backend,
        model=args.model,
        storage=storage_settings(args, sqlite_default=SQLITE_PATH),
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
