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
    """Persistence seam for cached masked-logits results.

    Implementations (:class:`FileCacheStore`, :class:`NullCacheStore`) back
    :class:`CachedConnector`; ``get``/``put`` are the only operations used.
    """

    def get(self, *, model: str, backend: str, sequence: str) -> SequenceLogits | None:
        """Returns cached logits for (model, backend, sequence), or ``None`` on a miss."""
        ...

    def put(self, result: SequenceLogits, *, model: str, backend: str) -> None:
        """Stores ``result`` under its (model, backend, sequence) key."""
        ...


def _cache_key(model: str, backend: str, sequence: str) -> str:
    """Content-addressed key for one (model, backend, sequence) inference result.

    blake2b is fixed across processes, unlike the salted builtin ``hash()``,
    so a recompute overwrites the same cache entry rather than fragmenting.
    """
    # blake2b is fixed across processes, unlike the salted builtin hash.
    digest = hashlib.blake2b(
        f"{model}|{backend}|{sequence}".encode(), digest_size=16
    ).hexdigest()
    return digest


def _write_entry(
    entry_dir: Path, result: SequenceLogits, *, model: str, backend: str
) -> None:
    """Persists one cache entry: compressed logits plus a JSON sidecar.

    The vocab is stored in the sidecar so reads never need to import the
    tokenizer; the directory is created with ``exist_ok`` because a cache
    overwrites in place on recompute.
    """
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
    """Rehydrates a :class:`SequenceLogits` from a cache entry directory."""
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
        """Returns the per-result directory under the cache root."""
        return self._root / _cache_key(model, backend, sequence)

    def get(self, *, model: str, backend: str, sequence: str) -> SequenceLogits | None:
        """Returns the cached result if present, otherwise ``None`` (a miss)."""
        entry_dir = self._entry_dir(model=model, backend=backend, sequence=sequence)
        if not entry_dir.is_dir():
            return None
        return _read_entry(entry_dir)

    def put(self, result: SequenceLogits, *, model: str, backend: str) -> None:
        """Writes ``result`` to its content-addressed directory, overwriting any prior entry."""
        _write_entry(
            self._entry_dir(model=model, backend=backend, sequence=result.sequence),
            result,
            model=model,
            backend=backend,
        )


class NullCacheStore:
    """No-op store that turns CachedConnector into a transparent passthrough."""

    def get(self, *, model: str, backend: str, sequence: str) -> SequenceLogits | None:
        """Always returns ``None`` so the connector recomputes every call."""
        return None

    def put(self, result: SequenceLogits, *, model: str, backend: str) -> None:
        """Discards the result; nothing is ever stored."""


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
        """Wraps ``inner`` so its results flow through ``store``.

        ``model`` and ``backend`` tag every entry so caches stay
        backend-comparable: a result from one backend never satisfies a query
        for another.
        """
        self._inner = inner
        self._store = store
        self._model = model
        self._backend = backend
        self._last_cache_hit = False

    @property
    def last_cache_hit(self) -> bool:
        """Whether the most recent :meth:`masked_sequence_logits` call served a hit."""
        return self._last_cache_hit

    def peak_memory_bytes(self) -> int | None:
        """Delegates to the inner connector's peak-memory report, if any."""
        return self._inner.peak_memory_bytes()

    def masked_sequence_logits(self, sequence: str) -> SequenceLogits:
        """Serves a cache hit, or computes via the inner connector and stores it.

        Built by :func:`esmlab.connectors.get_connector`; the analysis pipeline
        calls only this method, so cache behavior is transparent to callers.
        Records hit/miss in :attr:`last_cache_hit` for the perf report.
        """
        cached = self._store.get(
            model=self._model, backend=self._backend, sequence=sequence
        )
        if cached is not None:
            self._last_cache_hit = True
            return cached
        self._last_cache_hit = False
        result = self._inner.masked_sequence_logits(sequence)
        self._store.put(result, model=self._model, backend=self._backend)
        return result
