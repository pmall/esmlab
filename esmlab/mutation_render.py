"""Report presentation: stored entries in, self-contained HTML pages out.

Every function here returns a ``str`` or a JSON-ready ``dict`` and never
touches the filesystem, so the same two calls back the files written by
:mod:`esmlab.mutation_report` and a future HTTP handler reading the same
storage.

A page is a static template plus one injected JSON payload: the templates in
``esmlab/templates`` hold all the markup and all the drawing code, and Python's
whole job is to produce the payload. Charts are drawn client-side by Chart.js
from a CDN, so nothing here renders images and the report stage needs no
plotting dependency.
"""

import json
from importlib import resources
from typing import Any

import numpy as np
import numpy.typing as npt

from esmlab.amino_acids import AA_TO_ISOELECTRIC_POINT, VALID_AMINO_ACIDS
from esmlab.mutation_scoring import (
    EntryAnalysis,
    rank_substitutions,
    uniform_entropy_bits,
)
from esmlab.storage import StoredLogits, logits_key

# A page's data is plain JSON, so the payload is typed as loosely as it is
# consumed: the template reads it dynamically and so do the tests.
Payload = dict[str, Any]

PAYLOAD_MARKER = "__PAYLOAD__"

# Reference lines drawn across the entropy chart: the entropy of a uniform
# choice among this many amino acids, so a bar's height reads as "this position
# is about as free as picking among N residues".
UNIFORM_REFERENCE_SIZES = (16, 8, 4, 2)

# How many extremes the page's highlights section lists per category.
HIGHLIGHT_COUNT = 5

# Percentile of |LLR| the substitution matrix's color scale saturates at.
# Scaling to the maximum instead lets one catastrophic substitution - they run
# to tens of nats where the bulk of the matrix sits under one - compress every
# other cell into the neutral midpoint, so the matrix reads as blank. Cells
# past this magnitude are drawn at full color and their exact value stays in
# the cell's tooltip.
LLR_COLOR_PERCENTILE = 98.0

# Heatmap rows run from most basic to most acidic, so the charge gradient is
# visible down the column rather than scattered through alphabetical order.
AA_BY_ISOELECTRIC_POINT = tuple(
    sorted(
        AA_TO_ISOELECTRIC_POINT,
        key=lambda aa: AA_TO_ISOELECTRIC_POINT[aa],
        reverse=True,
    )
)


def entry_payload(
    entry: StoredLogits,
    analysis: EntryAnalysis,
    *,
    threshold: float,
    top_k: int,
) -> Payload:
    """All the data one entry's page draws, as plain JSON-ready values.

    This is the contract between the analysis and the page: the template reads
    nothing else, so it is also the response body a web view would serve. The
    LLR matrix is emitted with its columns already re-ordered to
    :data:`AA_BY_ISOELECTRIC_POINT` - the row order is a presentation choice,
    and doing it here keeps the template free of amino-acid knowledge.

    One row per scored residue, and nothing is sliced here: a record is only
    ever masked over the region its header named, so the analysis already
    covers exactly what the page shows. The page numbers those residues 1..n -
    the sub-sequence is the subject - while ``full_sequence`` with ``start`` and
    ``stop`` carry the context it sits in, exactly as storage holds them.
    """
    llr = analysis.llr
    residues = analysis.residues
    columns = [VALID_AMINO_ACIDS.index(aa) for aa in AA_BY_ISOELECTRIC_POINT]
    best_columns = llr.argmax(axis=1)
    return {
        "key": logits_key(entry.logits.sequence, analysis.start, analysis.stop),
        "label": entry.label,
        "model": entry.model,
        "created_utc": entry.created_utc,
        "metadata": entry.metadata,
        "full_sequence": entry.logits.sequence,
        "start": analysis.start,
        "stop": analysis.stop,
        "sequence": residues,
        "length": len(residues),
        "threshold": threshold,
        "positions": list(range(1, len(residues) + 1)),
        "wildtype": list(residues),
        "entropy": _floats(analysis.entropies),
        "entropy_references": [
            {"label": f"{size} AA uniform", "bits": uniform_entropy_bits(size)}
            for size in UNIFORM_REFERENCE_SIZES
        ],
        "deleterious_fraction": _floats(analysis.fractions),
        "tolerant": [bool(fraction < threshold) for fraction in analysis.fractions],
        "amino_acids": list(AA_BY_ISOELECTRIC_POINT),
        "llr": [_floats(row) for row in llr[:, columns]],
        "llr_color_scale": _color_scale(llr),
        "llr_color_percentile": LLR_COLOR_PERCENTILE,
        "llr_max_abs": round(float(np.abs(llr).max(initial=0.0)), 4),
        "top_alt": [VALID_AMINO_ACIDS[column] for column in best_columns],
        "top_alt_llr": _floats(llr[np.arange(len(llr)), best_columns]),
        "top_substitutions": [
            {"position": position, "wt": wildtype, "alt": alternative, "llr": score}
            for position, wildtype, alternative, score in rank_substitutions(
                llr, residues, top_k
            )
        ],
        "most_constrained": _extremes(analysis.entropies, residues),
        "most_tolerant": _extremes(analysis.fractions, residues),
    }


def index_payload(entries: list[Payload]) -> Payload:
    """One model's listing page: a row per rendered entry, newest first.

    Takes the entry payloads themselves rather than a separate summary so the
    listing and the pages it links cannot disagree about what was rendered.
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
        }
        for payload in entries
    ]
    rows.sort(key=lambda row: row["created_utc"], reverse=True)
    return {"model": entries[0]["model"] if entries else "", "entries": rows}


def render_entry(payload: Payload) -> str:
    """One entry's complete, self-contained HTML page."""
    return _fill("report.html", payload)


def render_index(payload: Payload) -> str:
    """The listing page linking to every entry page in the same directory."""
    return _fill("index.html", payload)


def _fill(template: str, payload: Payload) -> str:
    """Substitutes the payload into a template's JSON island.

    ``</`` is escaped because the payload lands inside a ``<script>`` element,
    where that sequence would otherwise end the element early - a label or a
    metadata value is arbitrary text and may contain one.
    """
    markup = resources.files("esmlab.templates").joinpath(template).read_text()
    encoded = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    return markup.replace(PAYLOAD_MARKER, encoded)


def _color_scale(llr: npt.NDArray[np.float64]) -> float:
    """The |LLR| the matrix's color ramp saturates at.

    A high percentile rather than the maximum, so the readable range covers the
    bulk of the matrix instead of one outlier. Degenerate matrices - an all-zero
    one, or one whose percentile lands on zero - fall back to the maximum and
    then to a positive epsilon, because the page divides by this.
    """
    magnitudes = np.abs(llr)
    scale = float(np.percentile(magnitudes, LLR_COLOR_PERCENTILE)) if llr.size else 0.0
    if scale <= 0.0:
        scale = float(magnitudes.max(initial=0.0))
    return round(scale, 4) if scale > 0.0 else float(np.finfo(np.float32).eps)


def _floats(values: npt.NDArray[np.float64]) -> list[float]:
    """Rounds an array to plain floats, short enough to keep the payload small.

    Six decimals is far beyond what the charts and tables display, so this is
    lossless as far as the page is concerned.
    """
    return [round(float(value), 6) for value in values]


def _extremes(values: npt.NDArray[np.float64], sequence: str) -> list[Payload]:
    """The :data:`HIGHLIGHT_COUNT` lowest-scoring positions, as page rows.

    Both highlight lists want the same shape - lowest entropy is the most
    constrained position, lowest deleterious fraction the most tolerant - so
    they share one helper.
    """
    order = np.argsort(values)[:HIGHLIGHT_COUNT]
    return [
        {
            "position": int(index) + 1,
            "wt": sequence[int(index)],
            "value": round(float(values[index]), 4),
        }
        for index in order
    ]
