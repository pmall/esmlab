"""Matplotlib renderings of the mutation-analysis outputs (PNG files only)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from esmlab.amino_acids import AA_TO_ISOELECTRIC_POINT, VALID_AMINO_ACIDS
from esmlab.mutation_scoring import uniform_entropy_bits


def plot_entropy(
    entropies: npt.NDArray[np.float64], sequence: str, out_path: Path
) -> Path:
    """Per-position entropy bars with uniform-distribution reference lines."""
    positions = np.arange(1, len(entropies) + 1)
    plt.figure(figsize=(16, 4))
    plt.bar(positions, entropies, color="skyblue")
    for alphabet_size in (16, 8, 4, 2):
        plt.axhline(
            uniform_entropy_bits(alphabet_size),
            color="grey",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7 - 0.15 * int(np.log2(alphabet_size)),
            label=f"{alphabet_size} AA uniform",
        )
    plt.xlabel("Position")
    plt.ylabel("Entropy (bits)")
    plt.title(f"Entropy per position ({len(sequence)} residues)")
    step = max(1, len(entropies) // 20)
    plt.xticks(positions[::step])
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_deleterious_fraction(
    fractions: npt.NDArray[np.float64], threshold: float, out_path: Path
) -> Path:
    """Scatter of the fraction of deleterious substitutions at each position.

    Positions below the dashed line tolerate enough substitutions to make
    them candidate sites for library design.
    """
    positions = np.arange(1, len(fractions) + 1)
    plt.figure(figsize=(16, 4))
    plt.scatter(positions, fractions, s=10)
    plt.axhline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"threshold {threshold}",
    )
    plt.ylim(0, 1)
    plt.xlabel("Position")
    plt.ylabel("Fraction deleterious (LLR < 0)")
    plt.title("Fraction of deleterious substitutions per position")
    step = max(1, len(fractions) // 20)
    plt.xticks(positions[::step], fontsize=8)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_llr_heatmap(
    llr: npt.NDArray[np.float64], sequence: str, out_path: Path
) -> Path:
    """All single-substitution LLRs; rows are amino acids by descending pI."""
    aa_by_pI = sorted(
        AA_TO_ISOELECTRIC_POINT,
        key=lambda aa: AA_TO_ISOELECTRIC_POINT[aa],
        reverse=True,
    )
    # LLR columns follow VALID_AMINO_ACIDS order; rows are re-ordered by pI.
    image = llr[:, [VALID_AMINO_ACIDS.index(aa) for aa in aa_by_pI]].T

    magnitude = float(np.abs(image).max())
    vmax = magnitude if magnitude > 0 else float(np.finfo(np.float32).eps)

    fig, ax = plt.subplots(figsize=(16, 4))
    mesh = ax.imshow(
        image,
        aspect="auto",
        cmap="bwr_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_yticks(np.arange(len(aa_by_pI)), labels=aa_by_pI)
    ax.set_xlabel("Sequence position")
    ax.set_ylabel("Substituted amino acid")
    ax.set_title("Zero-shot substitution log-likelihood ratios")

    length = image.shape[1]
    ticks = [tick - 1 for tick in range(1, length + 1, 5)]
    if ticks[-1] != length - 1:
        ticks.append(length - 1)
    ax.set_xticks(ticks, labels=[str(tick + 1) for tick in ticks], fontsize=8)
    fig.colorbar(mesh, ax=ax, label="LLR vs wildtype (nats)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
