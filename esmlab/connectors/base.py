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
    """Masked-language-model logits for one sequence.

    ``logits[i]`` holds the token-distribution row predicted when residue i of
    ``sequence`` was the masked position; BOS/EOS rows are stripped so axis 0
    maps one-to-one onto sequence residues. ``vocab`` maps token strings to
    column indices of the logits array.
    """

    sequence: str
    logits: npt.NDArray[np.float32]
    vocab: Mapping[str, int]


class ModelConnector(Protocol):
    """Backend contract: mask each residue, read the logits, plus memory reporting.

    Every backend (stub, local, biohub, modal) implements
    :meth:`masked_sequence_logits`. Connectors only compute: persisting the
    result is the caller's job, via :mod:`esmlab.storage`.
    :meth:`peak_memory_bytes` exposes the peak memory of the last
    call where the backend can observe it (CUDA on the local backend); it
    returns ``None`` for backends with no observable memory (stub, biohub, modal,
    or the local backend on CPU).
    """

    def masked_sequence_logits(self, sequence: str) -> SequenceLogits:
        """Returns per-position masked-logits for ``sequence`` (axis 0 aligns to residues)."""
        ...

    def peak_memory_bytes(self) -> int | None:
        """Peak memory of the last :meth:`masked_sequence_logits` call, or ``None``.

        ``None`` means the backend cannot observe memory (stub, biohub, modal,
        or the local backend on CPU). The local backend on CUDA returns
        ``torch.cuda.max_memory_allocated`` reset around each call.
        """
        ...
