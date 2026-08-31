"""Single entry point for building backend connectors."""

from esmlab.connectors.base import (
    CANONICAL_SEQUENCE_MODELS,
    ModelConnector,
    ParamSpec,
)
from esmlab.connectors.biohub import PARAMS as _BIOHUB_PARAMS
from esmlab.connectors.local import PARAMS as _LOCAL_PARAMS
from esmlab.connectors.modal_app import PARAMS as _MODAL_PARAMS
from esmlab.connectors.stub import PARAMS as _STUB_PARAMS

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
    modal_gpu: str,
) -> ModelConnector:
    """Builds the connector selected by ``backend``.

    Each backend branch passes only its relevant parameters. Connectors only
    compute: persisting their results is the caller's job, via
    :mod:`esmlab.storage`.
    """
    if model not in CANONICAL_SEQUENCE_MODELS:
        raise ValueError(
            f"Unknown sequence model {model!r}; "
            f"expected one of {CANONICAL_SEQUENCE_MODELS}"
        )

    match backend:
        case "stub":
            from esmlab.connectors.stub import StubConnector

            return StubConnector(model=model)
        case "local":
            from esmlab.connectors.local import LocalConnector

            return LocalConnector(model=model, device=device, batch_size=batch_size)
        case "biohub":
            from esmlab.connectors.biohub import BiohubConnector

            return BiohubConnector(model=model, token=biohub_api_key)
        case "modal":
            from esmlab.connectors.modal_app import ModalConnector

            return ModalConnector(
                model=model,
                token_id=modal_token_id,
                token_secret=modal_token_secret,
                gpu=modal_gpu,
            )
        case _:
            raise ValueError(f"Unknown backend {backend!r}; expected one of {BACKENDS}")
