"""Single entry point for building backend connectors."""

from esmlab.connectors.base import CANONICAL_MODELS, ModelConnector

BACKENDS = ("stub", "local", "forge", "modal")


def get_connector(
    backend: str,
    model: str = "esmc-600m",
    *,
    device: str = "auto",
    batch_size: int = 32,
    api_key: str = "",
) -> ModelConnector:
    """Builds the connector selected by ``backend``.

    ``device``/``batch_size`` only apply to the local backend; ``api_key`` only
    to Forge. Kept in one factory so callers never import a concrete backend.
    """
    if model not in CANONICAL_MODELS:
        raise ValueError(f"Unknown model {model!r}; expected one of {CANONICAL_MODELS}")

    match backend:
        case "stub":
            from esmlab.connectors.stub import StubConnector

            return StubConnector(model=model)
        case "local":
            from esmlab.connectors.local import LocalConnector

            return LocalConnector(model=model, device=device, batch_size=batch_size)
        case "forge":
            from esmlab.connectors.forge import ForgeConnector

            return ForgeConnector(model=model, token=api_key)
        case "modal":
            from esmlab.connectors.modal_app import ModalConnector

            return ModalConnector(model=model)
        case _:
            raise ValueError(f"Unknown backend {backend!r}; expected one of {BACKENDS}")
