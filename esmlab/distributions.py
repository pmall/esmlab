"""Per-position amino-acid distributions read out of stored logits.

Topic-agnostic: every consumer of stored logits starts by turning a row of
token logits into a distribution over the 20 canonical amino acids, and the
entropy of that distribution is the one summary that needs no reference
residue. The peptides topic builds log-likelihood ratios on top of the same
rows (:mod:`esmlab.peptides_scoring`); the entropy topic
(:mod:`esmlab.sequences_scoring`) stops here.

All statistics are computed over the 20 canonical residues only: the token
distributions also cover special tokens (<cls>, <mask>, ...), which would
inflate entropies uniformly without carrying any residue information.
"""

import numpy as np
import numpy.typing as npt

from esmlab.amino_acids import VALID_AMINO_ACIDS
from esmlab.connectors.base import SequenceLogits


def uniform_entropy_bits(alphabet_size: int) -> float:
    """Entropy of a uniform distribution over ``alphabet_size`` symbols.

    Both report topics draw these as reference lines on an entropy chart, so a
    bar reads as "this position is about as free as picking among N residues".
    """
    return float(np.log2(alphabet_size))


def aa_log_probs(result: SequenceLogits) -> npt.NDArray[np.float64]:
    """Log-softmax restricted to the canonical amino acid columns, shape (L, 20).

    The foundation everything else is derived from: :func:`entropy_per_position`
    here, and :func:`~esmlab.peptides_scoring.llr_matrix` in the peptides topic.
    """
    columns = [result.vocab[aa] for aa in VALID_AMINO_ACIDS]
    rows = result.logits[:, columns].astype(np.float64)
    shifted = rows - rows.max(axis=1, keepdims=True)
    normalizer = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return shifted - normalizer


def entropy_per_position(result: SequenceLogits) -> npt.NDArray[np.float64]:
    """Shannon entropy in bits of the amino-acid distribution at each position.

    Low entropy is a position the model is confident about, whatever residue is
    actually there: unlike a log-likelihood ratio it needs no wildtype column,
    which is why it is the whole of the entropy topic's math.
    """
    probs = np.exp(aa_log_probs(result))
    return -(probs * np.log2(probs)).sum(axis=1)
