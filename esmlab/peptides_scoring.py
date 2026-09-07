"""Zero-shot mutation-scoring math over one peptide's masked-position logits.

Every function takes a :class:`SequenceLogits` or derived numpy arrays and is
pure CPU work, so it can be tested without any inference backend. The
distribution these numbers are read out of is
:mod:`esmlab.distributions`, which every logits topic shares; what belongs here
is what needs a wildtype residue to mean anything.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy import newaxis

from esmlab.amino_acids import VALID_AMINO_ACIDS
from esmlab.connectors.base import SequenceLogits
from esmlab.distributions import aa_log_probs, entropy_per_position


def llr_matrix(result: SequenceLogits) -> npt.NDArray[np.float64]:
    """Log-likelihood ratio of every substitution relative to the wildtype.

    Column j corresponds to VALID_AMINO_ACIDS[j]; the wildtype column is 0 by
    construction, positive values are tolerated improvements.

    Built on :func:`aa_log_probs`; consumed by
    :func:`deleterious_fraction_per_position`, :func:`rank_substitutions`,
    and the report's substitution matrix and summary CSV.
    """
    log_probs = aa_log_probs(result)
    # ``residues`` and not ``sequence``: a row is one masked position, and only
    # a request covering the whole sequence makes the two the same string.
    wildtype_columns = [VALID_AMINO_ACIDS.index(aa) for aa in result.residues]
    return (
        log_probs - log_probs[np.arange(len(log_probs)), wildtype_columns][:, newaxis]
    )


def deleterious_fraction_per_position(
    llr: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Fraction of the 19 non-wildtype substitutions with negative LLR.

    Takes the :func:`llr_matrix` output; feeds the report's per-position
    scatter chart and its per-position tolerant flag.
    """
    return (llr < 0).sum(axis=1) / (llr.shape[1] - 1)


def rank_substitutions(
    llr: npt.NDArray[np.float64], sequence: str, top_k: int
) -> list[tuple[int, str, str, float]]:
    """The ``top_k`` best-scoring substitutions as (position, wt, alt, llr).

    Positions are 1-indexed; wildtype self-substitutions are excluded because
    their LLR is 0 by definition and would otherwise crowd the ranking.

    Consumes :func:`llr_matrix`'s output; called by
    :mod:`esmlab.peptides_render` for the top-tolerated-substitutions list.
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


@dataclass(frozen=True)
class EntryAnalysis:
    """Everything derived from one stored entry, computed once.

    The three arrays here are what every consumer of this module needs, and
    they build on each other (``fractions`` is derived from ``llr``), so a
    caller that recomputed them separately would do the same work twice. Row
    ``i`` of each describes residue ``start + i`` of the stored sequence, whose
    wildtype is ``residues[i]``.

    Assembled by :func:`analyze`; consumed by :mod:`esmlab.peptides_render`,
    which turns it into the report payload.
    """

    residues: str
    start: int
    stop: int
    entropies: npt.NDArray[np.float64]
    llr: npt.NDArray[np.float64]
    fractions: npt.NDArray[np.float64]


def analyze(result: SequenceLogits) -> EntryAnalysis:
    """Runs the whole scoring chain over one sequence's logits.

    The single entry point into this module for the report stage: pure CPU
    work over stored arrays, with no model, no storage and no I/O.
    """
    llr = llr_matrix(result)
    return EntryAnalysis(
        residues=result.residues,
        start=result.start,
        stop=result.stop,
        entropies=entropy_per_position(result),
        llr=llr,
        fractions=deleterious_fraction_per_position(llr),
    )
