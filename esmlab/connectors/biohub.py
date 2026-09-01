"""Biohub Platform backend (hosted inference, requires a BIOHUB_API_KEY)."""

import numpy as np

from esmlab.connectors.base import SequenceLogits
from esmlab.params import ParamSpec

# Canonical model ids -> Biohub Platform model names.
BIOHUB_MODEL_NAMES = {
    "esmc-300m": "esmc-300m-2024-12",
    "esmc-600m": "esmc-600m-2024-12",
    "esmc-6b": "esmc-6b-2024-12",
}

PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec(
        flag="--biohub-api-key",
        dest="biohub_api_key",
        env="BIOHUB_API_KEY",
        help="Biohub Platform API token (defaults to $BIOHUB_API_KEY from .env)",
        required=True,
    ),
)


class BiohubConnector:
    """Masks residues with the "_" placeholder convention of the SDK.

    The leave-one-out requests are dispatched concurrently through the SDK's
    parallel executor, mirroring the official mutation-scoring tutorial.
    """

    def __init__(self, model: str, token: str, url: str = "https://biohub.ai") -> None:
        """Builds the SDK ``esmc_client`` pointed at the Biohub Platform hosted API.

        The canonical model id is mapped to its dated Biohub Platform model name via
        :data:`BIOHUB_MODEL_NAMES`. The SDK import is deferred so the package
        does not require Biohub Platform credentials unless this backend is selected.
        """
        from esm.sdk import esmc_client

        self._client = esmc_client(
            model=BIOHUB_MODEL_NAMES[model], url=url, token=token
        )

    def masked_sequence_logits(self, sequence: str) -> SequenceLogits:
        """Scores every leave-one-out mask via concurrent Biohub Platform requests.

        Each residue is replaced in turn by the SDK ``"_"`` mask placeholder;
        the resulting masked strings are dispatched in parallel through the
        SDK executor (the dispatch pattern comes from the official
        mutation-scoring tutorial, but nothing here is specific to it). Row
        ``i`` of variant ``i`` is the prediction at the masked residue (+1
        skips the BOS token). Returns a :class:`SequenceLogits`.
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

    def peak_memory_bytes(self) -> None:
        """Inference runs on the hosted Biohub Platform API; client-side memory is not observable."""
        return
