"""Biohub Platform backend (hosted inference, requires a BIOHUB_API_KEY)."""

import numpy as np

from esmlab.connectors.base import SequenceLogits, check_region
from esmlab.params import ParamSpec

# Canonical model ids -> Biohub Platform model names.
BIOHUB_MODEL_NAMES = {
    "esmc-300m": "esmc-300m-2024-12",
    "esmc-600m": "esmc-600m-2024-12",
    "esmc-6b": "esmc-6b-2024-12",
}

# Most masked residues this backend will score in one call. Every mask is a
# separate full-length request, so a masked sweep costs L*L tokens for a
# length-L sequence: a 500-residue protein is a third of the daily credit
# budget, spent before anything comes back. A peptide never needs more than a
# handful of masks, so a request for more is a whole-sequence job pointed at
# the wrong entrypoint - use sequence_logits, which is one request.
MAX_MASKED_RESIDUES = 20

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

    def masked_sequence_logits(
        self, sequence: str, start: int, stop: int
    ) -> SequenceLogits:
        """Scores the region's leave-one-out masks via concurrent Platform requests.

        Refuses a region longer than :data:`MAX_MASKED_RESIDUES` before making
        any request, because the cost is quadratic and paid before anything
        comes back.

        Each residue of ``start``..``stop`` is replaced in turn by the SDK
        ``"_"`` mask placeholder;
        the resulting masked strings are dispatched in parallel through the
        SDK executor (the dispatch pattern comes from the official
        mutation-scoring tutorial, but nothing here is specific to it). Row
        ``i`` of variant ``i`` is the prediction at the masked residue (+1
        skips the BOS token). Returns a :class:`SequenceLogits`.
        """
        # Both guards run before the deferred imports: rejecting a bad request
        # should not cost a torch import.
        check_region(sequence, start, stop)
        if stop - start + 1 > MAX_MASKED_RESIDUES:
            raise ValueError(
                f"region {start}-{stop} asks for {stop - start + 1} masks, over "
                f"the Biohub Platform limit of {MAX_MASKED_RESIDUES}: each mask "
                f"is a separate full-length request, so this would cost "
                f"{(stop - start + 1) * len(sequence)} tokens. Use "
                "sequence_logits for a whole sequence, or score a shorter region."
            )

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

        scored = range(start, stop + 1)
        masked = [
            sequence[: position - 1] + "_" + sequence[position:] for position in scored
        ]
        with parallel_executor(show_progress=False) as executor:
            outputs = executor.execute_batch(user_func=fetch_logits, sequence=masked)

        # The executor returns failures in place rather than raising, so a
        # refusal would otherwise surface as an attribute error on the first
        # "logits" read rather than as what the Platform actually said.
        failures = [output for output in outputs if isinstance(output, Exception)]
        if failures:
            raise RuntimeError(
                f"{len(failures)}/{len(outputs)} Biohub Platform requests failed; "
                f"first failure: {failures[0]}"
            )

        # Row i of variant i is the prediction at its masked residue; +1 skips BOS.
        rows = torch.stack(
            [
                output.logits.sequence[position]
                for position, output in zip(scored, outputs)
            ]
        )
        from esm.tokenization import get_esmc_model_tokenizers

        return SequenceLogits(
            sequence=sequence,
            start=start,
            stop=stop,
            logits=rows.float().cpu().numpy().astype(np.float32),
            vocab=get_esmc_model_tokenizers().get_vocab(),
        )

    def sequence_logits(self, sequence: str) -> SequenceLogits:
        """Scores the whole sequence with one unmasked request.

        One request instead of one per residue, and no executor to manage:
        the Platform bills per token, so this is the difference between L and
        L*L tokens for a length-L sequence - about 0.03 credits for a 300-mer
        against 9. Rows ``1..L`` of the response are the residues; row 0 is
        BOS.
        """
        import torch
        from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig
        from esm.tokenization import get_esmc_model_tokenizers

        # The SDK returns refusals rather than raising them, here as in the
        # masked path; one request means there is no batch to summarize.
        protein_tensor = self._client.encode(ESMProtein(sequence=sequence))
        if isinstance(protein_tensor, ESMProteinError):
            raise protein_tensor
        output = self._client.logits(protein_tensor, LogitsConfig(sequence=True))
        if isinstance(output, ESMProteinError):
            raise output

        sequence_logits = output.logits.sequence if output.logits else None
        assert sequence_logits is not None, "LogitsConfig(sequence=True) returned none"
        # Row 0 is BOS and row L+1 is EOS; the residues sit between them.
        rows: torch.Tensor = sequence_logits[0, 1 : len(sequence) + 1]
        return SequenceLogits(
            sequence=sequence,
            start=1,
            stop=len(sequence),
            logits=rows.float().cpu().numpy().astype(np.float32),
            vocab=get_esmc_model_tokenizers().get_vocab(),
        )

    def peak_memory_bytes(self) -> None:
        """Inference runs on the hosted Biohub Platform API; client-side memory is not observable."""
        return
