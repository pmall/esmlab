"""Report stage: turn stored logits into one HTML page per stored entry.

Reads a :class:`~esmlab.storage.LogitsStorage` and never constructs a
connector, so this stage needs no credentials, no GPU and no model. Re-running
it with a different threshold or top-k is pure CPU work over arrays that are
already stored.

The whole stage is scoring plus writing: :func:`~esmlab.mutation_scoring.analyze`
derives the numbers and :mod:`esmlab.mutation_render` turns them into pages,
neither of which touches the filesystem. This module owns every path, which is
why the same two calls can back a web view over the same storage.

Reports are located by the sequence's storage key, not by its label: two input
records sharing a FASTA header are the same analysis if they are the same
sequence. Each model's ``index.html`` maps its keys back to labels for humans.
"""

from dataclasses import dataclass
from pathlib import Path

from esmlab.mutation_render import (
    Payload,
    entry_payload,
    index_payload,
    render_entry,
    render_index,
)
from esmlab.mutation_scoring import analyze
from esmlab.storage import LogitsStorage, StorageSettings, open_storage

INDEX_FILE = "index.html"


@dataclass(frozen=True)
class ReportSettings:
    """Fully-resolved configuration for one :func:`run_report` invocation.

    Built by the ``mutation_report`` script. ``model`` narrows the run to one
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
    ``<out_dir>/<model>/<key>.html`` - the model is in the path because the
    storage key is the sequence's alone, so two models would otherwise
    overwrite each other - beside an ``index.html`` listing that model's pages.
    Returns the list of written artifact paths.

    Raises :class:`ValueError` when the selection matches no stored entry,
    rather than leaving an empty report directory behind.
    """
    storage = open_storage(settings.storage)

    artifacts: list[Path] = []
    for model in _selected_models(settings, storage):
        model_dir = settings.out_dir / model
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


def _selected_models(settings: ReportSettings, storage: LogitsStorage) -> list[str]:
    """Which models this run renders, or a :class:`ValueError` explaining why none.

    ``--model`` is optional because reporting is cheap and the store is small:
    the useful default is "render everything you have". Asking for a model with
    nothing stored is still an error, since it is almost always a compute run
    that used a different one.
    """
    stored = storage.models()
    if not stored:
        raise ValueError(
            "Storage holds no entries at all, so there is nothing to report"
        )

    names = [name for name, _ in stored]
    if settings.model is None:
        return names
    if settings.model not in names:
        raise ValueError(_nothing_stored(settings.model, stored))
    return [settings.model]


def _nothing_stored(model: str, stored: list[tuple[str, int]]) -> str:
    """The message for a ``--model`` that matched no stored entry.

    Names what the store does hold, because the useful next command is the same
    one with a different ``--model`` or with none at all.
    """
    held = ", ".join(f"{name} ({count})" for name, count in stored)
    return (
        f"Storage holds no entries for model {model!r}; it holds: {held}. "
        f"Re-run with a --model it holds, or without --model for all of them."
    )
