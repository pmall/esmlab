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
# Default Modal GPU. ESMC runs bf16 on CUDA (see project.md, "Accelerated GPU
# kernels"), so any override must be an Ampere-or-newer card (SM >= 8.0):
# A10G, L4, A100, H100. The prebuilt flash-attn wheel is also SM 8.0-9.0 only.
_DEFAULT_MODAL_GPU = "H100"

# HuggingFace cache path inside the container (Modal runs as root). A
# per-model Modal Volume is mounted here so `from_pretrained` downloads each
# checkpoint once ever, not once per cold container. See project.md.
_HF_CACHE_DIR = "/root/.cache/huggingface"


def _hf_cache_volume_name(model: str) -> str:
    """Modal Volume name holding the HuggingFace cache for one model id."""
    return f"esmlab-hf-{model}"

# Keep in lockstep with the `esm` pin in pyproject.toml / the references/esm
# submodule commit.
_ESM_GIT = (
    "esm @ git+https://github.com/evolutionaryscale/esm.git"
    "@43ccece2ad485f27db46afdb67da2a9601e8f106"
)
# EvolutionaryScale's prebuilt flash-attn (py312 / pytorch 2.11 / CUDA 13,
# SM 8.0-9.0). Mirrors the `flash-attn` source in pyproject.toml.
_FLASH_ATTN_WHEEL = (
    "https://github.com/evolutionaryscale/wheels/releases/download/"
    "py312-pt211-cu13-sm80-90/"
    "flash_attn-2.7.4.post1-cp312-cp312-linux_x86_64.whl"
)

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
    ParamSpec(
        flag="--modal-gpu",
        dest="modal_gpu",
        env="MODAL_GPU",
        help=(
            f"Modal GPU type, Ampere or newer (default: $MODAL_GPU or "
            f"{_DEFAULT_MODAL_GPU}); e.g. A10G, L4, A100, A100-80GB, H100"
        ),
        default=_DEFAULT_MODAL_GPU,
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


def _build_app(gpu: str, model: str):
    """Constructs the Modal ``App``/image and the remote ``masked_logits`` worker.

    The image installs ``esm`` plus its fused CUDA kernels and this repo's
    ``esmlab`` source so the worker can call :func:`_connector_for`. The worker
    is pinned to one ``model`` (so its HuggingFace-cache Volume is per-model)
    and runs on a ``gpu`` GPU with a 5-minute scaledown window to keep the
    checkpoint warm between calls. Returns ``(app, masked_logits)`` for the
    caller to run.
    """
    import modal

    # ESMC only reaches its fused kernels on CUDA when the matching packages are
    # importable (see project.md, "Accelerated GPU kernels"). esm's own base
    # dependency set does not include them - and its published xformers wheel is
    # ABI-broken against the pinned torch and shadows flash-attn in the kernel
    # dispatch - so the image installs flash-attn + transformer-engine and drops
    # xformers. The CUDA *devel* base gives nvcc for the transformer-engine
    # build; that layer is cached, so the ~15 min compile happens once.
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12"
        )
        .env({"NVTE_FRAMEWORK": "pytorch", "HF_HOME": _HF_CACHE_DIR})
        .pip_install(_ESM_GIT, _FLASH_ATTN_WHEEL)
        .run_commands("python -m pip uninstall -y xformers")
        .pip_install("transformer-engine[pytorch]==2.15.0")
        .add_local_python_source("esmlab")
    )
    app = modal.App(name=f"{_MODAL_APP_NAME}-{model}", image=image)

    # Persistent per-model HuggingFace cache: `from_pretrained` populates it on
    # the first cold container, every later container (warm or cold) reads it.
    # Modal background-commits Volume writes on container shutdown.
    hf_cache = modal.Volume.from_name(
        _hf_cache_volume_name(model), create_if_missing=True
    )

    @app.function(
        gpu=gpu, scaledown_window=300, volumes={_HF_CACHE_DIR: hf_cache}
    )
    def masked_logits(sequence: str) -> _RemoteLogits:
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

    def __init__(
        self,
        model: str,
        *,
        token_id: str,
        token_secret: str,
        gpu: str = _DEFAULT_MODAL_GPU,
    ) -> None:
        """Stores the model id and GPU, and publishes Modal credentials to the env.

        The credentials are written to ``os.environ`` because Modal's client
        reads them on connect, which happens inside the lazy ``import modal``
        within :meth:`masked_sequence_logits`. ``gpu`` is the Modal GPU type
        the remote worker runs on.
        """
        self._model = model
        self._gpu = gpu or _DEFAULT_MODAL_GPU
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
        app, masked_logits = _build_app(self._gpu, self._model)
        with app.run():
            payload: _RemoteLogits = masked_logits.remote(sequence)
        return SequenceLogits(
            sequence=payload.sequence,
            logits=np.asarray(payload.logits, dtype=np.float32),
            vocab=payload.vocab,
        )

    def peak_memory_bytes(self) -> None:
        """Inference runs on a rented remote GPU; client-side memory is not observable."""
        return
