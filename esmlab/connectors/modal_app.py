"""Modal GPU backend: runs the local `esm` code on rented Modal GPUs.

Not exercised yet in this repo (needs a Modal account). The worker mirrors
`local.py` on a GPU container and keeps one loaded checkpoint per container.
`modal` is imported lazily so it is not required for the other backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from esmlab.connectors.base import SequenceLogits

if TYPE_CHECKING:
    from esmlab.connectors.local import LocalConnector

_MODAL_APP_NAME = "esmlab-esmc"
_MODAL_GPU = "A10G"


@dataclass(frozen=True)
class _RemoteLogits:
    sequence: str
    logits: npt.NDArray[np.float32]
    vocab: dict[str, int]


# Populated once per Modal container; the checkpoint then stays warm across calls.
_connectors: dict[str, LocalConnector] = {}


def _connector_for(model: str) -> LocalConnector:
    if model not in _connectors:
        from esmlab.connectors.local import LocalConnector

        _connectors[model] = LocalConnector(model=model, device="cuda")
    return _connectors[model]


def _build_app():
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
        result = _connector_for(model).masked_sequence_logits(sequence)
        return _RemoteLogits(
            sequence=result.sequence,
            logits=result.logits,
            vocab=dict(result.vocab),
        )

    return app, masked_logits


class ModalConnector:
    """Calls the remote worker through the connector interface."""

    def __init__(self, model: str) -> None:
        self._model = model

    def masked_sequence_logits(self, sequence: str) -> SequenceLogits:
        app, masked_logits = _build_app()
        with app.run():
            payload: _RemoteLogits = masked_logits.remote(sequence, model=self._model)
        return SequenceLogits(
            sequence=payload.sequence,
            logits=np.asarray(payload.logits, dtype=np.float32),
            vocab=payload.vocab,
        )
