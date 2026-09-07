"""Build whole-sequence entropy reports from logits a compute script stored.

A consuming phase over what ``sequences_logits.py`` stored in
``data/sequences.sqlite``: it reads a storage and never constructs a connector,
so it needs no credentials, no GPU and no model. Re-running with a different
``--window`` is pure CPU work over arrays that are already stored.

What a page says: one Shannon entropy per position, in bits, of ESMC's
amino-acid distribution there - how much freedom the model leaves at that
residue, against the 4.32-bit ceiling of a uniform choice among the 20. Unlike
the peptide report it needs no wildtype column, so it carries one number per
position and no substitution matrix, which is what keeps it readable for a
whole protein. Each page draws the entropy track with a rolling mean over
``--window`` positions, the distribution of those entropies, the extreme
positions at both ends, and a per-position table; the index ranks the model's
sequences by mean entropy, which is the comparison the topic exists for.

It iterates the storage rather than an input file — the store is the source of
truth for what this topic has computed — and renders every entry held for
``--model``. The store is the scope: a run reports on the sequences in
``--sqlite-path`` and on nothing else.

Where it writes: ``<out>/sequences/<model>/<key>.html``, one self-contained page
per entry, beside that model's ``index.html``. ``--out`` is the reports root
every topic shares (default ``data/reports``); the topic and the model are in
the path below it because a page is named after the entry's storage digest
alone, so neither another topic nor another model may share a directory with
it. Open a model's ``index.html`` to read a run; a page needs a network
connection the first time, for the charting library it loads from a CDN.

``--model`` is optional: the default renders every model the store holds, since
reporting costs no model time. Naming a model with nothing stored is a CLI
error listing the models that do have entries.

Takes the same storage flags as ``sequences_logits.py``, and reads by default
the same ``data/sequences.sqlite`` it writes.
"""

import argparse
import sys
from pathlib import Path

from esmlab.connectors.base import CANONICAL_SEQUENCE_MODELS
from esmlab.params import load_env
from esmlab.sequences_report import ReportSettings, run_report
from esmlab.storage import add_storage_arguments, storage_settings

SUMMARY = __doc__.partition("\n\n")[0] if __doc__ else ""

# The store this topic writes and reads. One database per topic, so a report
# covers this topic's runs and nothing else.
SQLITE_PATH = Path("data/sequences.sqlite")

DEFAULT_WINDOW = 9


def _build_parser() -> argparse.ArgumentParser:
    """Builds the argparse parser.

    No backend flags and no model credentials: this stage reads stored arrays
    and never constructs a connector. The storage flags are the same ones the
    compute scripts declare, so a run's printed command line is copyable.
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
    add_storage_arguments(parser, sqlite_default=SQLITE_PATH)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/reports"),
        help="Root for report artifacts, one directory per topic (default: data/reports)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=(
            "Positions in the rolling-mean window drawn over the entropy track "
            f"(default: {DEFAULT_WINDOW}; 1 draws no smoothing)"
        ),
    )
    return parser


def _report(args: argparse.Namespace) -> int:
    """Logic for one run: resolve storage params and render every stored entry.

    Loads ``.env`` so the storage env fallbacks resolve, assembles a
    :class:`ReportSettings`, calls :func:`run_report`, and prints the written
    artifacts. Returns the process exit code.
    """
    load_env()
    if args.window < 1:
        raise ValueError("--window must be at least 1 position")
    settings = ReportSettings(
        model=args.model,
        storage=storage_settings(args, sqlite_default=SQLITE_PATH),
        out_dir=args.out,
        window=args.window,
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
