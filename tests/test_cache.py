from pathlib import Path

import numpy as np

from esmlab.cache import CachedConnector, FileCacheStore, NullCacheStore
from esmlab.connectors.base import SequenceLogits
from esmlab.connectors.stub import STUB_VOCAB, StubConnector

SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"


def _result() -> SequenceLogits:
    """Builds a stubbed :class:`SequenceLogits` for the shared test sequence."""
    return StubConnector("esmc-600m").masked_sequence_logits(SEQUENCE)


def test_file_cache_store_miss_then_round_trip(tmp_path: Path) -> None:
    """A miss returns None; after put, the entry round-trips with equal logits/vocab/sequence."""
    store = FileCacheStore(tmp_path)
    assert store.get(model="esmc-600m", backend="stub", sequence=SEQUENCE) is None

    store.put(_result(), model="esmc-600m", backend="stub")
    loaded = store.get(model="esmc-600m", backend="stub", sequence=SEQUENCE)
    assert loaded is not None
    np.testing.assert_array_equal(loaded.logits, _result().logits)
    assert loaded.sequence == SEQUENCE
    assert loaded.vocab == dict(STUB_VOCAB)


def test_file_cache_store_keys_on_model_backend_and_sequence(tmp_path: Path) -> None:
    """Cache hits require an exact (model, backend, sequence) match."""
    store = FileCacheStore(tmp_path)
    store.put(_result(), model="esmc-600m", backend="stub")

    assert store.get(model="esmc-600m", backend="stub", sequence=SEQUENCE) is not None
    # A different model, backend, or sequence must miss, so backends stay comparable.
    assert store.get(model="esmc-300m", backend="stub", sequence=SEQUENCE) is None
    assert store.get(model="esmc-600m", backend="local", sequence=SEQUENCE) is None
    assert store.get(model="esmc-600m", backend="stub", sequence=SEQUENCE[1:]) is None


def test_file_cache_store_put_overwrites_idempotently(tmp_path: Path) -> None:
    """Writing the same entry twice is idempotent and still loads afterwards."""
    store = FileCacheStore(tmp_path)
    store.put(_result(), model="esmc-600m", backend="stub")
    store.put(_result(), model="esmc-600m", backend="stub")
    loaded = store.get(model="esmc-600m", backend="stub", sequence=SEQUENCE)
    assert loaded is not None


def test_null_cache_store_never_returns_or_stores(tmp_path: Path) -> None:
    """The null store always misses and stores nothing."""
    store = NullCacheStore()
    assert store.get(model="esmc-600m", backend="stub", sequence=SEQUENCE) is None
    store.put(_result(), model="esmc-600m", backend="stub")
    assert store.get(model="esmc-600m", backend="stub", sequence=SEQUENCE) is None


class _CountingConnector:
    """Wraps the stub and counts calls to observe cache hits versus misses."""

    def __init__(self) -> None:
        self._inner = StubConnector("esmc-600m")
        self.calls = 0

    def masked_sequence_logits(self, sequence: str) -> SequenceLogits:
        """Delegates to the stub while incrementing the call counter."""
        self.calls += 1
        return self._inner.masked_sequence_logits(sequence)


def test_cached_connector_stores_miss_and_serves_hit(tmp_path: Path) -> None:
    """A miss calls the inner connector; the next call is served from the cache without re-calling."""
    inner = _CountingConnector()
    connector = CachedConnector(
        inner=inner, store=FileCacheStore(tmp_path), model="esmc-600m", backend="stub"
    )

    first = connector.masked_sequence_logits(SEQUENCE)
    assert inner.calls == 1
    second = connector.masked_sequence_logits(SEQUENCE)
    # Served from the cache; the inner connector is not called again.
    assert inner.calls == 1
    np.testing.assert_array_equal(first.logits, second.logits)


def test_cached_connector_with_null_store_always_recomputes() -> None:
    """With a null store every call recomputes via the inner connector."""
    inner = _CountingConnector()
    connector = CachedConnector(
        inner=inner, store=NullCacheStore(), model="esmc-600m", backend="stub"
    )
    connector.masked_sequence_logits(SEQUENCE)
    connector.masked_sequence_logits(SEQUENCE)
    assert inner.calls == 2
