"""Caching of masked-logits results as a transparent connector decorator.

``CachedConnector`` wraps any :class:`ModelConnector`: it asks a
:class:`CacheStore` before calling the inner connector and stores the result
afterwards. ``NullCacheStore`` makes the wrapper a no-op passthrough, so the
same code path serves cached and uncached runs with no backend special-casing.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from esmlab.connectors.base import ModelConnector, ParamSpec, SequenceLogits

LOGITS_FILE = "logits.npz"
META_FILE = "meta.json"

CACHE_PARAMS: dict[str, tuple[ParamSpec, ...]] = {
    "null": (),
    "file": (
        ParamSpec(
            flag="--cache-root",
            dest="cache_root",
            env="",
            help="Directory for cached logits (required with --cache file)",
            type=Path,
            required=True,
        ),
    ),
}
CACHES = tuple(CACHE_PARAMS.keys())
ALL_CACHE_PARAMS: tuple[ParamSpec, ...] = tuple(
    spec for specs in CACHE_PARAMS.values() for spec in specs
)


class CacheStore(Protocol):
    """Persistence seam for cached masked-logits results."""

    def get(
        self, *, model: str, backend: str, sequence: str
    ) -> SequenceLogits | None: ...

    def put(self, result: SequenceLogits, *, model: str, backend: str) -> None: ...


def _cache_key(model: str, backend: str, sequence: str) -> str:
    # blake2b is fixed across processes, unlike the salted builtin hash.
    digest = hashlib.blake2b(
        f"{model}|{backend}|{sequence}".encode(), digest_size=16
    ).hexdigest()
    return digest


def _write_entry(
    entry_dir: Path, result: SequenceLogits, *, model: str, backend: str
) -> None:
    # A cache overwrites in place on recompute; exist_ok is intentional.
    entry_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(entry_dir / LOGITS_FILE, logits=result.logits)
    payload = {
        "name": entry_dir.name,
        "backend": backend,
        "model": model,
        "created_utc": datetime.now(UTC).isoformat(),
        "sequence": result.sequence,
        # Vocab is stored so reads never need to import the tokenizer.
        "vocab": dict(result.vocab),
    }
    (entry_dir / META_FILE).write_text(json.dumps(payload))


def _read_entry(entry_dir: Path) -> SequenceLogits:
    payload = json.loads((entry_dir / META_FILE).read_text())
    logits: npt.NDArray[np.float32] = np.load(entry_dir / LOGITS_FILE)["logits"]
    return SequenceLogits(
        sequence=payload["sequence"],
        logits=logits,
        vocab=payload["vocab"],
    )


class FileCacheStore:
    """Filesystem, content-addressed cache under a single root directory."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def _entry_dir(self, *, model: str, backend: str, sequence: str) -> Path:
        return self._root / _cache_key(model, backend, sequence)

    def get(self, *, model: str, backend: str, sequence: str) -> SequenceLogits | None:
        entry_dir = self._entry_dir(model=model, backend=backend, sequence=sequence)
        if not entry_dir.is_dir():
            return None
        return _read_entry(entry_dir)

    def put(self, result: SequenceLogits, *, model: str, backend: str) -> None:
        _write_entry(
            self._entry_dir(model=model, backend=backend, sequence=result.sequence),
            result,
            model=model,
            backend=backend,
        )


class NullCacheStore:
    """No-op store that turns CachedConnector into a transparent passthrough."""

    def get(self, *, model: str, backend: str, sequence: str) -> SequenceLogits | None:
        return None

    def put(self, result: SequenceLogits, *, model: str, backend: str) -> None:
        pass


class CachedConnector:
    """Decorates a ModelConnector with a CacheStore: serve hits, store misses."""

    def __init__(
        self,
        inner: ModelConnector,
        store: CacheStore,
        *,
        model: str,
        backend: str,
    ) -> None:
        self._inner = inner
        self._store = store
        self._model = model
        self._backend = backend

    def masked_sequence_logits(self, sequence: str) -> SequenceLogits:
        cached = self._store.get(
            model=self._model, backend=self._backend, sequence=sequence
        )
        if cached is not None:
            return cached
        result = self._inner.masked_sequence_logits(sequence)
        self._store.put(result, model=self._model, backend=self._backend)
        return result
