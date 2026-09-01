import numpy as np
import pytest

from esmlab.amino_acids import VALID_AMINO_ACIDS
from esmlab.connectors.stub import STUB_VOCAB, StubConnector

PETASE_FRAGMENT = "AADNPYQRGPDPTNASIEAATGPFAVGTQPIVG"


def test_stub_logits_are_deterministic_per_model_and_sequence() -> None:
    """The same (model, sequence) produces identical logits across calls."""
    first = StubConnector("esmc-600m").masked_sequence_logits(
        PETASE_FRAGMENT, 1, len(PETASE_FRAGMENT)
    )
    second = StubConnector("esmc-600m").masked_sequence_logits(
        PETASE_FRAGMENT, 1, len(PETASE_FRAGMENT)
    )
    np.testing.assert_array_equal(first.logits, second.logits)


def test_stub_logits_vary_between_sequences() -> None:
    """Different sequences seed different RNG draws, so logits differ."""
    first = StubConnector("esmc-600m").masked_sequence_logits(
        PETASE_FRAGMENT, 1, len(PETASE_FRAGMENT)
    )
    mutated = PETASE_FRAGMENT.replace("A", "G", 1)
    second = StubConnector("esmc-600m").masked_sequence_logits(mutated, 1, len(mutated))
    assert not np.array_equal(first.logits, second.logits)


def test_stub_shapes_and_vocab_cover_the_sequence() -> None:
    """Logits have shape (L, 20) float32 and the vocab is exactly the canonical amino acids."""
    result = StubConnector("esmc-300m").masked_sequence_logits(
        PETASE_FRAGMENT, 1, len(PETASE_FRAGMENT)
    )
    assert result.logits.shape == (len(PETASE_FRAGMENT), len(VALID_AMINO_ACIDS))
    assert result.logits.dtype == np.float32
    for aa in set(PETASE_FRAGMENT):
        assert aa in STUB_VOCAB
    assert sorted(STUB_VOCAB.values()) == list(range(len(VALID_AMINO_ACIDS)))


def test_stub_rows_are_valid_log_distributions() -> None:
    """Each row exponentiates to a normalized probability distribution (no -inf)."""
    result = StubConnector("esmc-600m").masked_sequence_logits(
        PETASE_FRAGMENT, 1, len(PETASE_FRAGMENT)
    )
    probs = np.exp(result.logits.astype(np.float64))
    assert np.all(probs.sum(axis=1) == pytest.approx(1.0))
    assert np.all(result.logits > -np.inf)


def test_stub_peak_memory_is_unobservable() -> None:
    """The stub allocates only small numpy arrays, so peak_memory_bytes is None."""
    assert StubConnector("esmc-600m").peak_memory_bytes() is None


def test_scoring_a_region_computes_only_its_rows() -> None:
    """Only the requested positions are scored, and each row is unchanged.

    The saving is the whole point of a region, and it is only sound if a row
    computed on its own equals the one a full sweep produces - the sequence is
    always passed whole, so the context behind each row is identical.
    """
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    connector = StubConnector("esmc-600m")

    everything = connector.masked_sequence_logits(sequence, 1, len(sequence))
    window = connector.masked_sequence_logits(sequence, 5, 7)

    assert (window.start, window.stop) == (5, 7)
    assert window.logits.shape == (3, everything.logits.shape[1])
    assert window.sequence == sequence
    assert window.residues == sequence[4:7]
    assert np.array_equal(window.logits, everything.logits[4:7])


def test_a_region_past_the_sequence_is_rejected() -> None:
    """Coordinates are checked against the sequence they name."""
    with pytest.raises(ValueError, match="outside 1-5"):
        StubConnector("esmc-600m").masked_sequence_logits("ACDEF", 4, 9)
