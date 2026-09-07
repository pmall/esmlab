"""Whole-sequence entropy math over stored logits: how constrained each position is.

The whole topic is one number per position - the Shannon entropy of the
model's amino-acid distribution there, from :mod:`esmlab.distributions` - plus
the summaries that make a sequence comparable to another. Nothing here needs a
wildtype residue, so unlike the peptides topic there is no substitution matrix
and no reference column: a low-entropy position is one ESMC is confident about
whatever residue happens to sit there.

Pure CPU work over arrays that are already stored, so it runs with no model, no
storage and no I/O.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from esmlab.connectors.base import SequenceLogits
from esmlab.distributions import entropy_per_position

# Bin width of the entropy histogram, in bits. The range is 0..log2(20), so
# this puts about seventeen bars under the distribution - enough to show
# whether a sequence is bimodal (a constrained core plus a free surface) rather
# than uniformly middling, which is the comparison the histogram exists for.
HISTOGRAM_BIN_BITS = 0.25


def rolling_mean(
    values: npt.NDArray[np.float64], window: int
) -> npt.NDArray[np.float64]:
    """Centred rolling mean of ``values`` over ``window`` positions.

    A per-position entropy track over a few hundred residues reads as noise;
    the smoothed line is what shows where the constrained stretches are. The
    window shrinks at the ends rather than emitting ``NaN``, so the returned
    array covers every position and the page can plot it against the same axis.

    A window of 1 or less returns the values unchanged.
    """
    if window <= 1 or values.size == 0:
        return values.astype(np.float64)
    half = window // 2
    smoothed = np.empty_like(values, dtype=np.float64)
    for index in range(values.size):
        low = max(0, index - half)
        high = min(values.size, index + half + 1)
        smoothed[index] = values[low:high].mean()
    return smoothed


def histogram(
    entropies: npt.NDArray[np.float64], limit: float
) -> list[tuple[float, int]]:
    """Counts of positions per entropy bin, as ``(bin start in bits, count)``.

    Bins are :data:`HISTOGRAM_BIN_BITS` wide and span ``0..limit``, which is
    fixed by the alphabet rather than by the data, so two sequences' histograms
    are drawn on the same axis and can be read against each other.
    """
    edges = np.arange(0.0, limit + HISTOGRAM_BIN_BITS, HISTOGRAM_BIN_BITS)
    counts, _ = np.histogram(entropies, bins=edges)
    return [(float(edge), int(count)) for edge, count in zip(edges, counts)]


@dataclass(frozen=True)
class SequenceAnalysis:
    """One stored entry's entropy track and the summary that ranks it.

    Row ``i`` of ``entropies`` and ``smoothed`` describes residue ``start + i``
    of the stored sequence, whose residue is ``residues[i]``.

    The scalars are what the index page sorts and compares sequences on:
    ``mean`` is the headline - the average number of bits of freedom ESMC
    leaves per residue - and the quartiles say whether that mean comes from a
    uniformly middling sequence or from a constrained core beside a free
    surface.

    Assembled by :func:`analyze`; consumed by :mod:`esmlab.sequences_render`.
    """

    residues: str
    start: int
    stop: int
    entropies: npt.NDArray[np.float64]
    smoothed: npt.NDArray[np.float64]
    mean: float
    median: float
    lower_quartile: float
    upper_quartile: float
    minimum: float
    maximum: float


def analyze(result: SequenceLogits, *, window: int) -> SequenceAnalysis:
    """Runs the whole entropy chain over one entry's logits.

    The single entry point into this module for the report stage. ``window`` is
    a presentation knob for :func:`rolling_mean` and changes nothing about the
    per-position entropies or the summary scalars.
    """
    entropies = entropy_per_position(result)
    quartiles = np.percentile(entropies, (25, 50, 75))
    return SequenceAnalysis(
        residues=result.residues,
        start=result.start,
        stop=result.stop,
        entropies=entropies,
        smoothed=rolling_mean(entropies, window),
        mean=float(entropies.mean()),
        median=float(quartiles[1]),
        lower_quartile=float(quartiles[0]),
        upper_quartile=float(quartiles[2]),
        minimum=float(entropies.min()),
        maximum=float(entropies.max()),
    )
