"""Modal GPU backend: runs the local `esm` code on rented Modal GPUs.

Not exercised yet in this repo (needs a Modal account). The worker mirrors
`local.py` on a GPU container and keeps one loaded checkpoint per container.
`modal` is imported lazily so it is not required for the other backends.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from esmlab.connectors.base import ParamSpec, SequenceLogits

if TYPE_CHECKING:
    from esmlab.connectors.local import LocalConnector

_MODAL_APP_NAME = "esmlab-esmc"
_MODAL_GPU = "A10G"

PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec(
        flag="--modal-token-id",
        dest="modal_token_id",
        env="MODAL_TOKEN_ID",
        help="Modal token id (defaults to $MODAL_TOKEN_ID from .env)",
        required=True,
    ),
    ParamSpec(
        flag="--modal-token-secret",
        dest="modal_token_secret",
        env="MODAL_TOKEN_SECRET",
        help="Modal token secret (defaults to $MODAL_TOKEN_SECRET from .env)",
        required=True,
    ),
)


@dataclass(frozen=True)
class _RemoteLogits:
    """Serializable :class:`SequenceLogits` payload crossing the Modal RPC boundary.

    Plain-dataclass twin of :class:`SequenceLogits` with a concrete ``dict``
    vocab so the result marshals back from the remote worker.
    """

    sequence: str
    logits: npt.NDArray[np.float32]
    vocab: dict[str, int]


# Populated once per Modal container; the checkpoint then stays warm across calls.
_connectors: dict[str, LocalConnector] = {}


def _connector_for(model: str) -> LocalConnector:
    """Returns the per-model :class:`LocalConnector`, creating it on first use.

    Cached in ``_connectors`` so a warm Modal container reuses the loaded
    checkpoint across calls instead of reloading on every request.
    """
    if model not in _connectors:
        from esmlab.connectors.local import LocalConnector

        _connectors[model] = LocalConnector(model=model, device="cuda")
    return _connectors[model]


def _build_app():
    """Constructs the Modal ``App``/image and the remote ``masked_logits`` worker.

    The image installs the upstream ``esm`` package plus this repo's
    ``esmlab`` source so the worker can call :func:`_connector_for`. The
    worker function runs on an A10G GPU with a 5-minute scaledown window to
    keep the checkpoint warm between calls. Returns ``(app, masked_logits)``
    for the caller to run.
    """
    import modal

    # The upstream package installs its own dependency set (torch, CUDA wheels);
    # that is what we want inside the GPU container.
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("git+https://github.com/evolutionaryscale/esm.git")
        .add_local_python_source("esmlab")
    )
    app = modal.App(name=_MODAL_APP_NAME, image=image)

    @app.function(gpu=_MODAL_GPU, scaledown_window=300)
    def masked_logits(sequence: str, model: str) -> _RemoteLogits:
        """Remote entrypoint: scores ``sequence`` on the GPU container."""
        result = _connector_for(model).masked_sequence_logits(sequence)
        return _RemoteLogits(
            sequence=result.sequence,
            logits=result.logits,
            vocab=dict(result.vocab),
        )

    return app, masked_logits


class ModalConnector:
    """Calls the remote worker through the connector interface."""

    def __init__(self, model: str, *, token_id: str, token_secret: str) -> None:
        """Stores the model id and publishes Modal credentials to the environment.

        The credentials are written to ``os.environ`` because Modal's client
        reads them on connect, which happens inside the lazy ``import modal``
        within :meth:`masked_sequence_logits`.
        """
        self._model = model
        # Modal's client reads these on connect; set them before the lazy
        # `import modal` inside masked_sequence_logits runs.
        if token_id:
            os.environ["MODAL_TOKEN_ID"] = token_id
        if token_secret:
            os.environ["MODAL_TOKEN_SECRET"] = token_secret

    def masked_sequence_logits(self, sequence: str) -> SequenceLogits:
        """Runs one remote scoring call and rehydrates the result locally.

        Builds the app via :func:`_build_app`, invokes the remote
        ``masked_logits`` worker on a rented GPU, and converts the
        :class:`_RemoteLogits` payload back into a :class:`SequenceLogits`
        for the analysis pipeline.
        """
        app, masked_logits = _build_app()
        with app.run():
            payload: _RemoteLogits = masked_logits.remote(sequence, model=self._model)
        return SequenceLogits(
            sequence=payload.sequence,
            logits=np.asarray(payload.logits, dtype=np.float32),
            vocab=payload.vocab,
        )

    def peak_memory_bytes(self) -> None:
        """Inference runs on a rented remote GPU; client-side memory is not observable."""
        return
