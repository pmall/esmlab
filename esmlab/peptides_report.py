"""Report stage: turn stored logits into one peptide page per stored entry.

Reads a :class:`~esmlab.storage.LogitsStorage` and never constructs a
connector, so this stage needs no credentials, no GPU and no model. Re-running
it with a different threshold or top-k is pure CPU work over arrays that are
already stored.

The whole stage is scoring plus writing: :func:`~esmlab.peptides_scoring.analyze`
derives the numbers and :mod:`esmlab.peptides_render` turns them into pages,
neither of which touches the filesystem. This module owns every path, which is
why the same two calls can back a web view over the same storage.

Reports are located by the sequence's storage key, not by its label: two input
records sharing a FASTA header are the same analysis if they are the same
sequence, and they land under this topic's own directory of the shared reports
root. Each model's ``index.html`` maps its keys back to labels for humans.
"""

from dataclasses import dataclass
from pathlib import Path

from esmlab.peptides_render import (
    Payload,
    entry_payload,
    index_payload,
    render_entry,
    render_index,
)
from esmlab.peptides_scoring import analyze
from esmlab.storage import StorageSettings, open_storage, select_models

INDEX_FILE = "index.html"

# This topic's directory under the shared reports root. A page is named after
# the entry it renders, so two topics pointed at one directory would overwrite
# each other's pages.
TOPIC = "peptides"


@dataclass(frozen=True)
class ReportSettings:
    """Fully-resolved configuration for one :func:`run_report` invocation.

    Built by the ``peptides_report`` script. ``model`` narrows the run to one
    model's entries, and ``None`` reports every model the store holds;
    ``threshold`` and ``top_k`` are pure presentation knobs, free to change
    without touching the model.
    """

    model: str | None
    storage: StorageSettings
    out_dir: Path
    threshold: float
    top_k: int


def run_report(settings: ReportSettings) -> list[Path]:
    """Writes one HTML page per stored entry, plus the index that links them.

    Iterates the storage rather than an input file: the store is the source of
    truth for what has been computed. Each entry lands at
    ``<out_dir>/peptides/<model>/<key>.html`` - the model is in the path because
    storage key is the sequence's alone, so two models would otherwise
    overwrite each other - beside an ``index.html`` listing that model's pages.
    Returns the list of written artifact paths.

    Raises :class:`ValueError` when the selection matches no stored entry,
    rather than leaving an empty report directory behind.
    """
    storage = open_storage(settings.storage)

    artifacts: list[Path] = []
    for model in select_models(storage, settings.model):
        model_dir = settings.out_dir / TOPIC / model
        model_dir.mkdir(parents=True, exist_ok=True)

        payloads: list[Payload] = []
        for entry in storage.entries(model=model):
            payload = entry_payload(
                entry,
                analyze(entry.logits),
                threshold=settings.threshold,
                top_k=settings.top_k,
            )
            payloads.append(payload)
            page_path = model_dir / f"{payload['key']}.html"
            page_path.write_text(render_entry(payload))
            artifacts.append(page_path)
            print(f"  {model} {entry.label} ({_scope(payload)}) -> {page_path}")

        index_path = model_dir / INDEX_FILE
        index_path.write_text(render_index(index_payload(payloads)))
        artifacts.append(index_path)

    return artifacts


def _scope(payload: Payload) -> str:
    """What a run's line says the page covers, in the sequence's own numbering."""
    return (
        f"residues {payload['start']}-{payload['stop']} "
        f"of {len(payload['full_sequence'])}"
    )
