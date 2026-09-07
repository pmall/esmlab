"""Report stage: turn stored logits into one whole-sequence page per entry.

Reads a :class:`~esmlab.storage.LogitsStorage` and never constructs a
connector, so this stage needs no credentials, no GPU and no model. Re-running
it with a different smoothing window is pure CPU work over arrays that are
already stored.

The whole stage is scoring plus writing: :func:`~esmlab.sequences_scoring.analyze`
derives the numbers and :mod:`esmlab.sequences_render` turns them into pages,
neither of which touches the filesystem. This module owns every path, which is
why the same two calls can back a web view over the same storage.

It reports on every entry the store holds, and the store is the scope: this
topic writes its own database, so a run covers the sequences that were computed
into it and nothing else. Reports are located by the sequence's storage key,
not by its label, and land under this topic's own directory of the shared
reports root, because a page is named after its entry and the peptide report
names its pages the same way.
"""

from dataclasses import dataclass
from pathlib import Path

from esmlab.rendering import Payload
from esmlab.sequences_render import (
    entry_payload,
    index_payload,
    render_entry,
    render_index,
)
from esmlab.sequences_scoring import analyze
from esmlab.storage import StorageSettings, open_storage, select_models

INDEX_FILE = "index.html"

# This topic's directory under the shared reports root. A page is named after
# the entry it renders, so two topics pointed at one directory would overwrite
# each other's pages.
TOPIC = "sequences"


@dataclass(frozen=True)
class ReportSettings:
    """Fully-resolved configuration for one :func:`run_report` invocation.

    Built by the ``sequences_report`` script. ``model`` narrows the run to one
    model's entries, and ``None`` reports every model the store holds;
    ``window`` is a pure presentation knob, free to change without touching the
    model.
    """

    model: str | None
    storage: StorageSettings
    out_dir: Path
    window: int


def run_report(settings: ReportSettings) -> list[Path]:
    """Writes one entropy page per stored entry, plus the index that ranks them.

    Iterates the storage rather than an input file: the store is the source of
    truth for what has been computed. Each entry lands at
    ``<out_dir>/sequences/<model>/<key>.html`` - the model is in the path because
    the storage key is the sequence's alone, so two models would otherwise
    overwrite each other - beside an ``index.html`` listing that model's pages
    by mean entropy. Returns the list of written artifact paths.

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
                entry, analyze(entry.logits, window=settings.window)
            )
            payloads.append(payload)
            page_path = model_dir / f"{payload['key']}.html"
            page_path.write_text(render_entry(payload))
            artifacts.append(page_path)
            print(
                f"  {model} {entry.label} ({payload['length']} residues, "
                f"mean {payload['summary']['mean']:.3f} bits) -> {page_path}"
            )

        index_path = model_dir / INDEX_FILE
        index_path.write_text(render_index(index_payload(payloads)))
        artifacts.append(index_path)

    return artifacts
