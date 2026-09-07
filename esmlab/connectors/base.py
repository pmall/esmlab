"""Connector contract and the canonical model identifiers every backend maps.

Backend CLI/env parameters live in :mod:`esmlab.params`, which storage
backends share.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

# Canonical model identifiers, grouped by the task family they serve. Each
# backend maps these to its own naming scheme (Biohub Platform names, HF repo
# ids). Sequence models answer masked-logits queries (ESMC); structure models
# predict 3D structure (ESMFold2, and its single-sequence "fast" variant).
CANONICAL_SEQUENCE_MODELS = ("esmc-300m", "esmc-600m", "esmc-6b")
CANONICAL_STRUCTURE_MODELS = ("esmfold2", "esmfold2-fast")

# How a set of rows was read out of the model. The two are different
# measurements, not two speeds for one measurement, so a stored entry says
# which one produced it and a report never mixes them.
#
# - "masked" replaces each scored residue with <mask> in turn, so a row is a
#   genuine prediction from context alone: L forward passes, and the entropy
#   is how constrained that site is. The peptides topic pays this, because a
#   peptide is short and a mutation score has to mean something per position.
# - "single-pass" runs the unmasked sequence once and keeps every row. The
#   model can see the residue it predicts, so the distribution leans toward
#   reproducing it and entropies come out slightly low; measured against
#   masked rows on three proteins (esmc-300m), mean entropy fell by 0.02-0.16
#   bits at a rank correlation of 0.84-0.96, with a fifth of positions moving
#   more than 0.5 bits on one of them. The sequences topic takes that trade:
#   whole proteins at one pass each instead of L, which is the difference
#   between scoring a proteome and scoring six sequences.
SCORING_METHODS = ("masked", "single-pass")


@dataclass(frozen=True)
class SequenceLogits:
    """Masked-language-model logits for residues ``start``..``stop`` of a sequence.

    ``logits[i]`` holds the token-distribution row predicted when residue
    ``start + i`` was masked; BOS/EOS rows are stripped. Coordinates are 1-based
    and inclusive.

    ``sequence`` is the whole sequence, not just the scored part: it is the
    context every row was predicted in, and the rows mean nothing without it.
    The same residues read out of two different sequences are two different
    results. Only ``start``..``stop`` are masked, so scoring a 15-residue
    peptide inside a 500-residue protein is 15 forward passes rather than 500.

    ``vocab`` maps token strings to column indices of the logits array.
    """

    sequence: str
    start: int
    stop: int
    logits: npt.NDArray[np.float32]
    vocab: Mapping[str, int]

    @property
    def residues(self) -> str:
        """The scored residues, one per row of ``logits``."""
        return self.sequence[self.start - 1 : self.stop]


def check_region(sequence: str, start: int, stop: int) -> None:
    """Rejects coordinates that do not name residues of ``sequence``.

    Shared by every backend so they agree on the 1-based inclusive convention
    and fail the same way.
    """
    if not 1 <= start <= stop <= len(sequence):
        raise ValueError(
            f"region {start}-{stop} outside 1-{len(sequence)} "
            "(coordinates are 1-based and inclusive)"
        )


class ModelConnector(Protocol):
    """Backend contract: the two ways to read logits, plus memory reporting.

    Every backend (stub, local, biohub, modal) implements both readouts.
    :meth:`masked_sequence_logits` masks the region asked for, one forward pass
    per residue; :meth:`sequence_logits` runs a single pass over the unmasked
    sequence and keeps every row. Which one a topic wants is a question about
    the number it reports, not about the backend - see
    :data:`SCORING_METHODS`. Connectors only compute: persisting the result is
    the caller's job, via :mod:`esmlab.storage`.
    :meth:`peak_memory_bytes` exposes the peak memory of the last
    call where the backend can observe it (CUDA on the local backend); it
    returns ``None`` for backends with no observable memory (stub, biohub, modal,
    or the local backend on CPU).
    """

    def masked_sequence_logits(
        self, sequence: str, start: int, stop: int
    ) -> SequenceLogits:
        """Masked logits for residues ``start``..``stop`` of ``sequence``.

        ``sequence`` is always passed whole - it is the context - while
        ``start``/``stop`` say which residues to mask and score.
        """
        ...

    def sequence_logits(self, sequence: str) -> SequenceLogits:
        """Single-pass logits for every residue of ``sequence``, nothing masked.

        One forward pass returns a distribution at every position at once, so
        the whole sequence costs one pass rather than one per residue. The
        model sees the residue it is predicting, which is what makes this an
        approximation rather than a cheaper route to the same numbers; see
        :data:`SCORING_METHODS`. The region is always ``1..len(sequence)``.
        """
        ...

    def peak_memory_bytes(self) -> int | None:
        """Peak memory of the last :meth:`masked_sequence_logits` call, or ``None``.

        ``None`` means the backend cannot observe memory (stub, biohub, modal,
        or the local backend on CPU). The local backend on CUDA returns
        ``torch.cuda.max_memory_allocated`` reset around each call.
        """
        ...
