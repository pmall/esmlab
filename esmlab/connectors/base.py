"""Connector contract shared by every inference backend."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

# Canonical model identifiers accepted by every backend factory; each backend
# maps these to its own naming scheme (Forge names, HF repo ids).
CANONICAL_MODELS = ("esmc-300m", "esmc-600m")


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
    """One method is all analysis needs: mask each residue, read the logits."""

    def masked_sequence_logits(self, sequence: str) -> SequenceLogits: ...
