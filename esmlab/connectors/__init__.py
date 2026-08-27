"""Single entry point for building backend connectors."""

from pathlib import Path
from typing import TYPE_CHECKING

from esmlab.connectors.base import CANONICAL_MODELS, ModelConnector, ParamSpec
from esmlab.connectors.biohub import PARAMS as _BIOHUB_PARAMS
from esmlab.connectors.local import PARAMS as _LOCAL_PARAMS
from esmlab.connectors.modal_app import PARAMS as _MODAL_PARAMS
from esmlab.connectors.stub import PARAMS as _STUB_PARAMS

if TYPE_CHECKING:
    # Imported lazily at runtime in get_connector to avoid a circular import
    # (cache.py imports from connectors.base, which triggers this __init__).
    from esmlab.cache import CachedConnector

BACKEND_PARAMS: dict[str, tuple[ParamSpec, ...]] = {
    "stub": _STUB_PARAMS,
    "local": _LOCAL_PARAMS,
    "biohub": _BIOHUB_PARAMS,
    "modal": _MODAL_PARAMS,
}
BACKENDS = tuple(BACKEND_PARAMS.keys())
ALL_BACKEND_PARAMS: tuple[ParamSpec, ...] = tuple(
    spec for specs in BACKEND_PARAMS.values() for spec in specs
)


def get_connector(
    backend: str,
    model: str = "esmc-600m",
    *,
    device: str,
    batch_size: int,
    biohub_api_key: str,
    modal_token_id: str,
    modal_token_secret: str,
    cache: str,
    cache_root: Path | None,
) -> "CachedConnector":
    """Builds the connector selected by ``backend``, always cache-wrapped.

    Each backend branch passes only its relevant parameters; the result is
    wrapped in a :class:`CachedConnector` with the store selected by ``cache``.
    No backend is special-cased — callers always receive a cache-decorated
    connector, which is why the return type is the concrete
    :class:`CachedConnector` rather than the bare :class:`ModelConnector`
    protocol.
    """
    # Imported lazily to avoid a circular import: cache.py imports from
    # connectors.base, which triggers this package's __init__.
    from esmlab.cache import CACHES, CachedConnector, FileCacheStore, NullCacheStore

    if model not in CANONICAL_MODELS:
        raise ValueError(f"Unknown model {model!r}; expected one of {CANONICAL_MODELS}")

    match backend:
        case "stub":
            from esmlab.connectors.stub import StubConnector

            inner: ModelConnector = StubConnector(model=model)
        case "local":
            from esmlab.connectors.local import LocalConnector

            inner = LocalConnector(model=model, device=device, batch_size=batch_size)
        case "biohub":
            from esmlab.connectors.biohub import BiohubConnector

            inner = BiohubConnector(model=model, token=biohub_api_key)
        case "modal":
            from esmlab.connectors.modal_app import ModalConnector

            inner = ModalConnector(
                model=model, token_id=modal_token_id, token_secret=modal_token_secret
            )
        case _:
            raise ValueError(f"Unknown backend {backend!r}; expected one of {BACKENDS}")

    match cache:
        case "null":
            store = NullCacheStore()
        case "file":
            # Guaranteed by resolve_params; narrows the type for pyright.
            assert cache_root is not None
            store = FileCacheStore(cache_root)
        case _:
            raise ValueError(f"Unknown cache {cache!r}; expected one of {CACHES}")

    return CachedConnector(inner=inner, store=store, model=model, backend=backend)
