"""Whole-sequence report presentation: stored entries in, self-contained pages out.

Every function here returns a ``str`` or a JSON-ready ``dict`` and never
touches the filesystem, so the same two calls back the files written by
:mod:`esmlab.sequences_report` and a future HTTP handler reading the same
storage. The template-filling mechanism itself is :mod:`esmlab.rendering`.

The pages carry one number per position and no substitution matrix, which is
what separates this topic from the peptide report over the very same stored
logits: an entropy page stays readable for a whole protein, where a 20-by-L
matrix does not.
"""

import numpy as np
import numpy.typing as npt

from esmlab.amino_acids import VALID_AMINO_ACIDS
from esmlab.distributions import uniform_entropy_bits
from esmlab.rendering import Payload, fill_template
from esmlab.sequences_scoring import SequenceAnalysis, histogram
from esmlab.storage import StoredLogits, logits_key

# Reference lines drawn across the entropy chart: the entropy of a uniform
# choice among this many amino acids, so a bar's height reads as "this position
# is about as free as picking among N residues".
UNIFORM_REFERENCE_SIZES = (16, 8, 4, 2)

# How many extremes the page's highlights section lists per category.
HIGHLIGHT_COUNT = 8

# The most entropy a position can carry: a uniform choice among the canonical
# residues. Every chart and every histogram is drawn against this ceiling
# rather than against the data, so two sequences are read on one scale.
ENTROPY_LIMIT = uniform_entropy_bits(len(VALID_AMINO_ACIDS))


def entry_payload(entry: StoredLogits, analysis: SequenceAnalysis) -> Payload:
    """All the data one entry's page draws, as plain JSON-ready values.

    This is the contract between the analysis and the page: the template reads
    nothing else, so it is also the response body a web view would serve.

    The page numbers the scored residues 1..n and draws nothing else: this
    topic's records cover their whole sequence, so there is no region to mark
    inside a context and a position is just a position. ``full_sequence`` with
    ``start`` and ``stop`` are still carried, exactly as storage holds them,
    because :func:`index_payload` reads them and they are the entry's
    provenance.
    """
    residues = analysis.residues
    return {
        "key": logits_key(entry.logits.sequence, analysis.start, analysis.stop),
        "label": entry.label,
        "backend": entry.backend,
        "model": entry.model,
        "created_utc": entry.created_utc,
        "metadata": entry.metadata,
        "full_sequence": entry.logits.sequence,
        "start": analysis.start,
        "stop": analysis.stop,
        "sequence": residues,
        "length": len(residues),
        "positions": list(range(1, len(residues) + 1)),
        "residues": list(residues),
        "entropy": _floats(analysis.entropies),
        "smoothed": _floats(analysis.smoothed),
        "entropy_limit": round(ENTROPY_LIMIT, 6),
        "entropy_references": [
            {"label": f"{size} AA uniform", "bits": uniform_entropy_bits(size)}
            for size in UNIFORM_REFERENCE_SIZES
        ],
        "histogram": [
            {"bits": round(edge, 4), "count": count}
            for edge, count in histogram(analysis.entropies, ENTROPY_LIMIT)
        ],
        "summary": _summary(analysis),
        "most_constrained": _extremes(analysis.entropies, residues, highest=False),
        "most_variable": _extremes(analysis.entropies, residues, highest=True),
    }


def index_payload(entries: list[Payload]) -> Payload:
    """One run's listing page: a row per rendered entry, most constrained first.

    Sorted by mean entropy rather than by time, because the listing is the
    comparison: reading one sequence against another is the reason to score a
    set of them. Takes the entry payloads themselves rather than a separate
    summary so the listing and the pages it links cannot disagree.
    """
    rows = [
        {
            "key": payload["key"],
            "label": payload["label"],
            "length": payload["length"],
            "start": payload["start"],
            "stop": payload["stop"],
            "full_length": len(payload["full_sequence"]),
            "created_utc": payload["created_utc"],
            **payload["summary"],
        }
        for payload in entries
    ]
    rows.sort(key=lambda row: row["mean"])
    return {
        "backend": entries[0]["backend"] if entries else "",
        "model": entries[0]["model"] if entries else "",
        "entropy_limit": round(ENTROPY_LIMIT, 6),
        "entries": rows,
    }


def render_entry(payload: Payload) -> str:
    """One entry's complete, self-contained HTML page."""
    return fill_template("sequences_entry.html", payload)


def render_index(payload: Payload) -> str:
    """The listing page linking to every entry page in the same directory."""
    return fill_template("sequences_index.html", payload)


def _summary(analysis: SequenceAnalysis) -> Payload:
    """The scalars both the entry page's header and the index's row show.

    One shape in both places, so a sequence's mean reads the same on its own
    page as in the table that ranks it against the others.
    """
    return {
        "mean": round(analysis.mean, 4),
        "median": round(analysis.median, 4),
        "lower_quartile": round(analysis.lower_quartile, 4),
        "upper_quartile": round(analysis.upper_quartile, 4),
        "minimum": round(analysis.minimum, 4),
        "maximum": round(analysis.maximum, 4),
    }


def _floats(values: npt.NDArray[np.float64]) -> list[float]:
    """Rounds an array to plain floats, short enough to keep the payload small.

    Six decimals is far beyond what the charts and tables display, so this is
    lossless as far as the page is concerned.
    """
    return [round(float(value), 6) for value in values]


def _extremes(
    values: npt.NDArray[np.float64], sequence: str, *, highest: bool
) -> list[Payload]:
    """The :data:`HIGHLIGHT_COUNT` most extreme positions, as page rows.

    Both highlight lists want the same shape - the lowest entropies are the
    positions ESMC is most certain about, the highest the ones it leaves most
    open - so they share one helper.
    """
    order = np.argsort(values)
    selected = order[::-1][:HIGHLIGHT_COUNT] if highest else order[:HIGHLIGHT_COUNT]
    return [
        {
            "position": int(index) + 1,
            "residue": sequence[int(index)],
            "value": round(float(values[index]), 4),
        }
        for index in selected
    ]
