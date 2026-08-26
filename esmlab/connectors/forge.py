"""Forge API backend (hosted inference, requires an ESM_API_KEY)."""

import numpy as np

from esmlab.connectors.base import SequenceLogits

# Canonical model ids -> Forge model names.
FORGE_MODEL_NAMES = {
    "esmc-300m": "esmc-300m-2024-12",
    "esmc-600m": "esmc-600m-2024-12",
}


class ForgeConnector:
    """Masks residues with the "_" placeholder convention of the SDK.

    The leave-one-out requests are dispatched concurrently through the SDK's
    parallel executor, mirroring the official mutation-scoring tutorial.
    """

    def __init__(self, model: str, token: str, url: str = "https://biohub.ai") -> None:
        from esm.sdk import esmc_client

        self._client = esmc_client(model=FORGE_MODEL_NAMES[model], url=url, token=token)

    def masked_sequence_logits(self, sequence: str) -> SequenceLogits:
        import torch
        from esm.sdk import parallel_executor
        from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig

        def fetch_logits(sequence: str):
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
