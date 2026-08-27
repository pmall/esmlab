"""Deterministic fake logits so the whole pipeline runs without any model."""

import hashlib

import numpy as np
import numpy.typing as npt

from esmlab.amino_acids import VALID_AMINO_ACIDS
from esmlab.connectors.base import ParamSpec, SequenceLogits

# Vocabulary of the stub backend: one column per canonical amino acid.
STUB_VOCAB: dict[str, int] = {aa: index for index, aa in enumerate(VALID_AMINO_ACIDS)}

PARAMS: tuple[ParamSpec, ...] = ()


def _stable_seed(model: str, sequence: str) -> int:
    """Deterministic 64-bit seed derived from the model id and sequence.

    ``hash()`` is salted per process, so this content hash guarantees the stub
    produces reproducible logits across runs for a given (model, sequence).
    Consumed by :meth:`StubConnector.masked_sequence_logits` to seed the RNG.
    """
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
        """Synthesizes one log-probability row per residue for ``sequence``.

        For each position a Dirichlet draw over the 20 amino acids is boosted
        toward the wildtype residue, producing varied entropies and
        deleterious fractions that exercise every branch of the analysis. The
        output is seeded from :func:`_stable_seed` so results are reproducible
        per (model, sequence) and feed directly into the pure-CPU math in
        :mod:`esmlab.mutation_scoring`. Satisfies the
        :class:`ModelConnector` protocol and is the default backend used by
        tests and the CLI when no model is available.
        """
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

    def peak_memory_bytes(self) -> None:
        """The stub allocates only small numpy arrays; no observable memory peak."""
        return
