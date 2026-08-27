"""Local inference backend running the `esm` package on this machine."""

import numpy as np

from esmlab.connectors.base import ParamSpec, SequenceLogits

# Canonical model ids -> HuggingFace repos published by EvolutionaryScale.
LOCAL_MODEL_REPOS = {
    "esmc-300m": "biohub/ESMC-300M",
    "esmc-600m": "biohub/ESMC-600M",
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

        The model id (``esmc-300m``/``esmc-600m``) maps to a HuggingFace repo
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

    def masked_sequence_logits(self, sequence: str) -> SequenceLogits:
        """Runs leave-one-out masking on the loaded checkpoint.

        Builds ``len(sequence)`` token-id copies, each with one residue
        replaced by the mask token (offset by +1 to skip the leading ``<cls>``),
        then runs forward passes in ``batch_size`` chunks and gathers the
        prediction row at each masked position. Returns a
        :class:`SequenceLogits` whose axis 0 maps one-to-one onto the
        sequence residues, consumed by :mod:`esmlab.mutation_scoring`.
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
        for start in range(0, sequence_length, self._batch_size):
            chunk = variants[start : start + self._batch_size]
            input_ids = torch.tensor(chunk, dtype=torch.long, device=self._device)
            with torch.inference_mode():
                output = self._model(input_ids=input_ids)
            positions = torch.arange(start, start + len(chunk)) + 1
            rows.append(output.logits[torch.arange(len(chunk)), positions])

        stacked = torch.cat(rows).float().cpu().numpy().astype(np.float32)
        return SequenceLogits(
            sequence=sequence,
            logits=stacked,
            vocab=self._tokenizer.get_vocab(),
        )
