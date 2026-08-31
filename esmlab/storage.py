"""Persistence of computed masked-logits, keyed by (model, sequence).

Storage is a layer of its own: connectors compute logits and never persist
them, and this module persists logits and never computes them. Callers
orchestrate the two — ``has`` before compute, ``save`` after — so neither side
knows about the other.

Every implementation binds to one resource (a directory here, a database
connection later), and that resource is the only scope: two stores pointed at
the same resource are the same store. Logits for a given ``(model, sequence)``
are the same artifact whichever backend produced them, so the backend is
deliberately absent from the key. The corollary is that a ``stub`` run must use
its own resource, or its fake logits will be served to a later real run.
"""

import hashlib
import json
import os
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from esmlab.connectors.base import SequenceLogits

LOGITS_FILE = "logits.npz"
META_FILE = "meta.json"


@dataclass(frozen=True)
class StoredLogits:
    """One persisted logits entry: the result plus its provenance.

    ``label`` is a display string carried alongside the entry (the FASTA header
    the sequence first arrived under). It is data, never identity: nothing
    looks an entry up by it, and two entries may share one.
    """

    model: str
    label: str
    created_utc: str
    logits: SequenceLogits


def logits_key(sequence: str) -> str:
    """Content-addressed key for one sequence.

    Hashes the sequence alone: the model already selects the directory, so
    folding it in would only make the same sequence look different under each
    model. Keeping it out means one sequence has one key everywhere, and
    ``<root>/*/<key>`` answers which models have scored it.

    blake2b is fixed across processes, unlike the salted builtin ``hash()``, so
    a recompute lands on the same entry instead of fragmenting the store.
    """
    return hashlib.blake2b(sequence.encode(), digest_size=16).hexdigest()


class LogitsStorage(Protocol):
    """Persistence seam for computed masked-logits.

    Implementations bind to one resource. :class:`FileLogitsStorage` is the
    filesystem one; a database-backed store would satisfy the same contract.
    """

    def has(self, *, model: str, sequence: str) -> bool:
        """Whether a complete entry exists, without deserializing it."""
        ...

    def load(self, *, model: str, sequence: str) -> StoredLogits:
        """Returns the stored entry; raises :class:`KeyError` when absent."""
        ...

    def save(self, result: SequenceLogits, *, model: str, label: str) -> None:
        """Persists ``result`` under ``(model, result.sequence)``."""
        ...

    def entries(self, *, model: str) -> Iterator[StoredLogits]:
        """Yields every stored entry for ``model``, in unspecified order."""
        ...


class FileLogitsStorage:
    """Filesystem storage under one root: ``<root>/<model>/<key>/``.

    ``model`` comes from the closed ``CANONICAL_SEQUENCE_MODELS`` set, so it is
    path-safe by construction and needs no sanitizing.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def _model_dir(self, model: str) -> Path:
        return self._root / model

    def _entry_dir(self, *, model: str, sequence: str) -> Path:
        return self._model_dir(model) / logits_key(sequence)

    def has(self, *, model: str, sequence: str) -> bool:
        """Whether a complete entry exists, without reading the logits array.

        Presence is defined by the sidecar rather than the directory because
        :meth:`save` writes the sidecar last: a directory holding only the npz
        is a torn write and must not count as a hit.
        """
        entry_dir = self._entry_dir(model=model, sequence=sequence)
        return (entry_dir / META_FILE).is_file()

    def load(self, *, model: str, sequence: str) -> StoredLogits:
        """Rehydrates the stored entry, raising :class:`KeyError` when absent."""
        entry_dir = self._entry_dir(model=model, sequence=sequence)
        if not (entry_dir / META_FILE).is_file():
            raise KeyError(f"No stored logits for {model!r} and this sequence")
        return _read_entry(entry_dir)

    def save(self, result: SequenceLogits, *, model: str, label: str) -> None:
        """Writes the entry atomically, overwriting any prior one.

        Persistence is unconditional, so a crash midway through a write would
        otherwise leave a half-entry that every later run skips forever. The
        payload is built in a sibling temporary directory and renamed into
        place, which is atomic within a filesystem.
        """
        entry_dir = self._entry_dir(model=model, sequence=result.sequence)
        staging = entry_dir.with_name(f".tmp-{entry_dir.name}-{os.getpid()}")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            _write_entry(staging, result, model=model, label=label)
            # rename onto an existing directory fails, so clear the old entry.
            shutil.rmtree(entry_dir, ignore_errors=True)
            staging.rename(entry_dir)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def entries(self, *, model: str) -> Iterator[StoredLogits]:
        """Yields every complete entry for ``model``; missing model dir yields nothing."""
        model_dir = self._model_dir(model)
        if not model_dir.is_dir():
            return
        for entry_dir in sorted(model_dir.iterdir()):
            if (entry_dir / META_FILE).is_file():
                yield _read_entry(entry_dir)


def _write_entry(
    entry_dir: Path, result: SequenceLogits, *, model: str, label: str
) -> None:
    """Writes the npz then the sidecar, in that order.

    The sidecar is last because :meth:`FileLogitsStorage.has` treats it as the
    completeness marker. The vocab lives in it so reads never import a
    tokenizer.
    """
    np.savez_compressed(entry_dir / LOGITS_FILE, logits=result.logits)
    payload = {
        "model": model,
        "label": label,
        "created_utc": datetime.now(UTC).isoformat(),
        "sequence": result.sequence,
        "vocab": dict(result.vocab),
    }
    (entry_dir / META_FILE).write_text(json.dumps(payload))


def _read_entry(entry_dir: Path) -> StoredLogits:
    """Rehydrates a :class:`StoredLogits` from an entry directory."""
    payload = json.loads((entry_dir / META_FILE).read_text())
    logits: npt.NDArray[np.float32] = np.load(entry_dir / LOGITS_FILE)["logits"]
    return StoredLogits(
        model=payload["model"],
        label=payload["label"],
        created_utc=payload["created_utc"],
        logits=SequenceLogits(
            sequence=payload["sequence"],
            logits=logits,
            vocab=payload["vocab"],
        ),
    )
