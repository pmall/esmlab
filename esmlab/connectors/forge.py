"""Forge API backend (hosted inference, requires a FORGE_API_KEY)."""

import numpy as np

from esmlab.connectors.base import ParamSpec, SequenceLogits

# Canonical model ids -> Forge model names.
FORGE_MODEL_NAMES = {
    "esmc-300m": "esmc-300m-2024-12",
    "esmc-600m": "esmc-600m-2024-12",
}

PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec(
        flag="--forge-api-key",
        dest="forge_api_key",
        env="FORGE_API_KEY",
        help="Forge API token (defaults to $FORGE_API_KEY from .env)",
        required=True,
    ),
)


class ForgeConnector:
    """Masks residues with the "_" placeholder convention of the SDK.

    The leave-one-out requests are dispatched concurrently through the SDK's
    parallel executor, mirroring the official mutation-scoring tutorial.
    """

    def __init__(self, model: str, token: str, url: str = "https://biohub.ai") -> None:
        """Builds the SDK ``esmc_client`` pointed at the Forge hosted API.

        The canonical model id is mapped to its dated Forge model name via
        :data:`FORGE_MODEL_NAMES`. The SDK import is deferred so the package
        does not require Forge credentials unless this backend is selected.
        """
        from esm.sdk import esmc_client

        self._client = esmc_client(model=FORGE_MODEL_NAMES[model], url=url, token=token)

    def masked_sequence_logits(self, sequence: str) -> SequenceLogits:
        """Scores every leave-one-out mask via concurrent Forge requests.

        Each residue is replaced in turn by the SDK ``"_"`` mask placeholder;
        the resulting masked strings are dispatched in parallel through the
        SDK executor (mirroring the official mutation-scoring tutorial). Row
        ``i`` of variant ``i`` is the prediction at the masked residue (+1
        skips the BOS token). Returns a :class:`SequenceLogits` for the
        :mod:`esmlab.mutation_scoring` layer.
        """
        import torch
        from esm.sdk import parallel_executor
        from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig

        def fetch_logits(sequence: str):
            """Encodes and scores one masked sequence; raises on SDK errors."""
            protein_tensor = self._client.encode(ESMProtein(sequence=sequence))
            if isinstance(protein_tensor, ESMProteinError):
                raise protein_tensor
            output = self._client.logits(protein_tensor, LogitsConfig(sequence=True))
            if isinstance(output, ESMProteinError):
                raise output
            return output

        masked = [
            sequence[:offset] + "_" + sequence[offset + 1 :]
            for offset in range(len(sequence))
        ]
        with parallel_executor(show_progress=False) as executor:
            outputs = executor.execute_batch(user_func=fetch_logits, sequence=masked)

        # Row i of variant i is the prediction at the masked residue; +1 skips BOS.
        rows = torch.stack(
            [
                output.logits.sequence[offset + 1]
                for offset, output in enumerate(outputs)
            ]
        )
        from esm.tokenization import get_esmc_model_tokenizers

        return SequenceLogits(
            sequence=sequence,
            logits=rows.float().cpu().numpy().astype(np.float32),
            vocab=get_esmc_model_tokenizers().get_vocab(),
        )
