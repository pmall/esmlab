"""Deterministic fake logits so the whole pipeline runs without any model."""

import hashlib

import numpy as np
import numpy.typing as npt

from esmlab.amino_acids import VALID_AMINO_ACIDS
from esmlab.connectors.base import SequenceLogits

# Vocabulary of the stub backend: one column per canonical amino acid.
STUB_VOCAB: dict[str, int] = {aa: index for index, aa in enumerate(VALID_AMINO_ACIDS)}


def _stable_seed(model: str, sequence: str) -> int:
    # hash() is salted per process, so use a content hash for reproducibility.
    digest = hashlib.blake2b(f"{model}|{sequence}".encode(), digest_size=8).digest()
    return int.from_bytes(digest)


class StubConnector:
    """Synthesizes plausible per-position distributions from a fixed seed.

    Each position gets a Dirichlet draw over the 20 amino acids with a random
    boost for the wildtype residue, producing varied entropies and
    deleterious fractions that exercise every branch of the analysis code.
    """

    def __init__(self, model: str) -> None:
        self._model = model

    def masked_sequence_logits(self, sequence: str) -> SequenceLogits:
        rng = np.random.default_rng(_stable_seed(self._model, sequence))
        rows: list[npt.NDArray[np.float32]] = []
        for wildtype in sequence:
            concentration = rng.uniform(0.3, 0.9)
            probs = rng.dirichlet(np.full(len(VALID_AMINO_ACIDS), concentration))
            probs[STUB_VOCAB[wildtype]] += rng.uniform(0.0, 3.0)
            probs /= probs.sum()
            rows.append(np.log(probs).astype(np.float32))
        return SequenceLogits(
            sequence=sequence,
            logits=np.stack(rows),
            vocab=dict(STUB_VOCAB),
        )
