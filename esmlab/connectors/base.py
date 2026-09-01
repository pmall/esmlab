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
    """Backend contract: mask each residue, read the logits, plus memory reporting.

    Every backend (stub, local, biohub, modal) implements
    :meth:`masked_sequence_logits`, masking only the region asked for.
    Connectors only compute: persisting the result is the caller's job, via
    :mod:`esmlab.storage`.
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

    def peak_memory_bytes(self) -> int | None:
        """Peak memory of the last :meth:`masked_sequence_logits` call, or ``None``.

        ``None`` means the backend cannot observe memory (stub, biohub, modal,
        or the local backend on CPU). The local backend on CUDA returns
        ``torch.cuda.max_memory_allocated`` reset around each call.
        """
        ...
