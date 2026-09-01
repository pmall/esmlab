"""Local inference backend running the `esm` package on this machine.

Fused CUDA kernels
------------------
ESMC runs pure-PyTorch by default and switches to fused CUDA kernels at
model-load time when the package is importable *and* the model is on CUDA. CPU
always uses the pure path (full fp32); its "missing kernel" warnings there are
expected and harmless. Two kernels matter, both CUDA-only:

- **transformer-engine** — fused LayerNorm+Linear/MLP with an fp32 reduction.
  Without it the bf16 LayerNorm drifts ~O(100) on the residual stream (it
  washes out after the final norm). Auto-enabled on CUDA when importable.
- **flash-attn** — FlashAttention-2 fused attention. Passing
  ``attn_implementation="sdpa"`` below does *not* disable it: for our unmasked
  fixed-length leave-one-out batches the dispatch picks flash-attn anyway.

Both are the ``gpu`` extra in ``pyproject.toml``, off by default because
transformer-engine compiles against the host CUDA toolkit at install time.
Enable with ``uv sync --extra gpu`` on a CUDA host (needs ``nvcc``);
:func:`_assert_fused_kernels_available` refuses a CUDA run missing either,
rather than running slow and drifting.

**GPU requirement:** this module loads the model in bfloat16 on any CUDA
device. Native bf16 and the prebuilt flash-attn wheel (SM 8.0-9.0) both need
**Ampere or newer** — A10G, L4, A100, H100. Turing (T4) and Volta (V100) have
no bf16 tensor cores and are not supported.

**xformers is removed.** esm hard-depends on it and its kernel dispatch tries
it *before* flash-attn, but its only published wheel bundles a torch-2.10
binary that cannot load against esm's pinned torch 2.11. The install still
succeeds and ``import xformers.ops`` still works (hollow), so esm sets
``XFORMERS_INSTALLED=True`` and shadows flash-attn. ``[tool.uv]
override-dependencies`` drops it with an always-false marker; flash-attn does
the same job.
"""

import numpy as np

from esmlab.connectors.base import SequenceLogits
from esmlab.params import ParamSpec

# Canonical model ids -> HuggingFace repos published by EvolutionaryScale.
LOCAL_MODEL_REPOS = {
    "esmc-300m": "biohub/ESMC-300M",
    "esmc-600m": "biohub/ESMC-600M",
    "esmc-6b": "biohub/ESMC-6B",
}

PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec(
        flag="--device",
        dest="device",
        env="",
        help="Torch device for the local backend (default: auto-detect)",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    ),
    ParamSpec(
        flag="--batch-size",
        dest="batch_size",
        env="",
        help="Masked variants per forward pass on the local backend",
        type=int,
        default=32,
    ),
)


def _assert_fused_kernels_available() -> None:
    """Fails loudly when a CUDA run is missing ESMC's fused kernels.

    On GPU the pure-PyTorch fallback is ~2-5x slower and drifts ~O(100) on the
    bf16 residual stream, so an accidentally-unoptimized container should be an
    error, not a silent slow run. Reads the import-time flags esm sets in
    :mod:`esm.models.esmc.kernels` (populated by installing the ``gpu`` extra:
    ``flash-attn`` and ``transformer-engine[pytorch]``; see this module's
    docstring).
    """
    from esm.models.esmc import kernels

    missing = []
    if not kernels.TE_INSTALLED:
        missing.append("transformer-engine (fused LayerNorm)")
    if not (kernels.XFORMERS_INSTALLED or kernels.FLASH_ATTN_INSTALLED):
        missing.append("flash-attn (fused attention)")
    if missing:
        raise RuntimeError(
            "CUDA inference requested but these fused kernels are not "
            f"importable: {', '.join(missing)}. Install the GPU extra with "
            "`uv sync --extra gpu` on a CUDA host (see this module's "
            "docstring, 'Fused CUDA kernels')."
        )


def resolve_device(device: str) -> str:
    """Maps the CLI device choice to a concrete torch device string.

    ``"auto"`` resolves to ``"cuda"`` when available else ``"cpu"``; the other
    choices pass through unchanged. Called from
    :meth:`LocalConnector.__init__`.
    """
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class LocalConnector:
    """Loads an ESMC checkpoint once, then answers masked-logits queries.

    All L leave-one-out variants of a sequence share one length, so they are
    stacked into a single token batch processed in ``batch_size`` chunks to
    bound forward-pass memory.
    """

    def __init__(self, model: str, device: str = "auto", batch_size: int = 32) -> None:
        """Loads the ESMC checkpoint once and keeps the tokenizer on device.

        The model id (``esmc-300m`` / ``esmc-600m`` / ``esmc-6b``) maps to a HF repo
        published by EvolutionaryScale. ``device`` is resolved via
        :func:`resolve_device`; ``batch_size`` bounds the forward-pass memory
        used by :meth:`masked_sequence_logits`. Heavy imports (torch, esm) are
        deferred so importing the package does not require the model stack.
        """
        import torch
        from esm.models.esmc import EsmcForMaskedLM
        from esm.tokenization import get_esmc_model_tokenizers

        self._torch = torch
        resolved_device = resolve_device(device)
        if resolved_device == "cuda":
            _assert_fused_kernels_available()
        # sdpa is the attention kernel available without flash-attn; bf16 only
        # pays off on accelerators, CPU keeps full precision.
        self._model = EsmcForMaskedLM.from_pretrained(
            LOCAL_MODEL_REPOS[model],
            device=resolved_device,
            dtype=None if resolved_device == "cpu" else torch.bfloat16,
            attn_implementation="sdpa",
        )
        self._device = resolved_device
        self._batch_size = batch_size
        self._tokenizer = get_esmc_model_tokenizers()
        self._last_peak_memory_bytes: int | None = None

    def masked_sequence_logits(self, sequence: str) -> SequenceLogits:
        """Runs leave-one-out masking on the loaded checkpoint.

        Builds ``len(sequence)`` token-id copies, each with one residue
        replaced by the mask token (offset by +1 to skip the leading ``<cls>``),
        then runs forward passes in ``batch_size`` chunks and gathers the
        prediction row at each masked position. Returns a
        :class:`SequenceLogits` whose axis 0 maps one-to-one onto the
        sequence residues. What consumes it is not this module's business:
        masked logits feed mutation scoring, embedding and classification
        alike.
        """
        torch = self._torch
        # Attribute access routes through BatchEncoding.__getattr__, avoiding
        # the imprecise subscript typing of its values.
        token_ids = [int(token_id) for token_id in self._tokenizer(sequence).input_ids]
        sequence_length = len(sequence)
        mask_id = self._tokenizer.mask_token_id
        assert mask_id is not None

        # Row i masks residue i; the +1 skips the leading <cls> token.
        variants = []
        for offset in range(1, sequence_length + 1):
            variant = token_ids.copy()
            variant[offset] = mask_id
            variants.append(variant)

        rows = []
        if self._device == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)
        for start in range(0, sequence_length, self._batch_size):
            chunk = variants[start : start + self._batch_size]
            input_ids = torch.tensor(chunk, dtype=torch.long, device=self._device)
            with torch.inference_mode():
                output = self._model(input_ids=input_ids)
            positions = torch.arange(start, start + len(chunk)) + 1
            rows.append(output.logits[torch.arange(len(chunk)), positions])
        if self._device == "cuda":
            self._last_peak_memory_bytes = int(
                torch.cuda.max_memory_allocated(self._device)
            )
        else:
            self._last_peak_memory_bytes = None

        stacked = torch.cat(rows).float().cpu().numpy().astype(np.float32)
        return SequenceLogits(
            sequence=sequence,
            logits=stacked,
            vocab=self._tokenizer.get_vocab(),
        )

    def peak_memory_bytes(self) -> int | None:
        """Peak CUDA memory of the last forward pass, or ``None`` on CPU.

        Set by :meth:`masked_sequence_logits` from
        ``torch.cuda.max_memory_allocated`` (reset around each call). Returns
        ``None`` on CPU, where there is no observable allocator.
        """
        return self._last_peak_memory_bytes
