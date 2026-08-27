"""Zero-shot mutation-analysis math operating on masked-position logits.

Every function takes a :class:`SequenceLogits` or derived numpy arrays and is
pure CPU work, so it can be tested without any inference backend. All amino
acid statistics are computed over the 20 canonical residues only: the token
distributions also cover special tokens (<cls>, <mask>, ...), which would
inflate entropies uniformly without carrying mutation information.
"""

import numpy as np
import numpy.typing as npt
from numpy import newaxis

from esmlab.amino_acids import VALID_AMINO_ACIDS
from esmlab.connectors.base import SequenceLogits


def uniform_entropy_bits(alphabet_size: int) -> float:
    """Entropy of a uniform distribution over ``alphabet_size`` symbols.

    Used by :mod:`esmlab.plotting` to draw uniform-distribution reference
    lines on the per-position entropy bar chart.
    """
    return float(np.log2(alphabet_size))


def aa_log_probs(result: SequenceLogits) -> npt.NDArray[np.float64]:
    """Log-softmax restricted to the canonical amino acid columns, shape (L, 20).

    The shared foundation for the rest of the module:
    :func:`entropy_per_position` and :func:`llr_matrix` both call this.
    """
    columns = [result.vocab[aa] for aa in VALID_AMINO_ACIDS]
    rows = result.logits[:, columns].astype(np.float64)
    shifted = rows - rows.max(axis=1, keepdims=True)
    normalizer = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return shifted - normalizer


def entropy_per_position(result: SequenceLogits) -> npt.NDArray[np.float64]:
    """Shannon entropy in bits of the amino-acid distribution at each position.

    Built on :func:`aa_log_probs`; consumed by :func:`run_analysis` for the
    entropy plot and the most-constrained-positions report.
    """
    probs = np.exp(aa_log_probs(result))
    return -(probs * np.log2(probs)).sum(axis=1)


def llr_matrix(result: SequenceLogits) -> npt.NDArray[np.float64]:
    """Log-likelihood ratio of every substitution relative to the wildtype.

    Column j corresponds to VALID_AMINO_ACIDS[j]; the wildtype column is 0 by
    construction, positive values are tolerated improvements.

    Built on :func:`aa_log_probs`; consumed by
    :func:`deleterious_fraction_per_position`, :func:`rank_substitutions`,
    and :func:`run_analysis` (heatmap + summary CSV).
    """
    log_probs = aa_log_probs(result)
    wildtype_columns = [VALID_AMINO_ACIDS.index(aa) for aa in result.sequence]
    return (
        log_probs - log_probs[np.arange(len(log_probs)), wildtype_columns][:, newaxis]
    )


def deleterious_fraction_per_position(
    llr: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Fraction of the 19 non-wildtype substitutions with negative LLR.

    Takes the :func:`llr_matrix` output; feeds
    :func:`tolerant_positions` and the per-position scatter plot in
    :func:`run_analysis`.
    """
    return (llr < 0).sum(axis=1) / (llr.shape[1] - 1)


def tolerant_positions(
    fractions: npt.NDArray[np.float64], threshold: float
) -> npt.NDArray[np.int64]:
    """Zero-based indices of positions whose deleterious fraction is below ``threshold``.

    Consumes :func:`deleterious_fraction_per_position`'s output; used by
    :func:`run_analysis` to list candidate library-design sites.
    """
    return np.flatnonzero(fractions < threshold)


def rank_substitutions(
    llr: npt.NDArray[np.float64], sequence: str, top_k: int
) -> list[tuple[int, str, str, float]]:
    """The ``top_k`` best-scoring substitutions as (position, wt, alt, llr).

    Positions are 1-indexed; wildtype self-substitutions are excluded because
    their LLR is 0 by definition and would otherwise crowd the ranking.

    Consumes :func:`llr_matrix`'s output; called by :func:`run_analysis` to
    print the top tolerated substitutions.
    """
    scores: list[tuple[int, str, str, float]] = []
    for position, wildtype in enumerate(sequence):
        wt_column = VALID_AMINO_ACIDS.index(wildtype)
        for alt_column, alternative in enumerate(VALID_AMINO_ACIDS):
            if alt_column == wt_column:
                continue
            scores.append(
                (position + 1, wildtype, alternative, float(llr[position, alt_column]))
            )
    scores.sort(key=lambda item: item[3], reverse=True)
    return scores[:top_k]
