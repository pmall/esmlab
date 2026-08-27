import numpy as np
import pytest

from esmlab.amino_acids import VALID_AMINO_ACIDS
from esmlab.connectors.stub import STUB_VOCAB, StubConnector

PETASE_FRAGMENT = "AADNPYQRGPDPTNASIEAATGPFAVGTQPIVG"


def test_stub_logits_are_deterministic_per_model_and_sequence() -> None:
    """The same (model, sequence) produces identical logits across calls."""
    first = StubConnector("esmc-600m").masked_sequence_logits(PETASE_FRAGMENT)
    second = StubConnector("esmc-600m").masked_sequence_logits(PETASE_FRAGMENT)
    np.testing.assert_array_equal(first.logits, second.logits)


def test_stub_logits_vary_between_sequences() -> None:
    """Different sequences seed different RNG draws, so logits differ."""
    first = StubConnector("esmc-600m").masked_sequence_logits(PETASE_FRAGMENT)
    second = StubConnector("esmc-600m").masked_sequence_logits(
        PETASE_FRAGMENT.replace("A", "G", 1)
    )
    assert not np.array_equal(first.logits, second.logits)


def test_stub_shapes_and_vocab_cover_the_sequence() -> None:
    """Logits have shape (L, 20) float32 and the vocab is exactly the canonical amino acids."""
    result = StubConnector("esmc-300m").masked_sequence_logits(PETASE_FRAGMENT)
    assert result.logits.shape == (len(PETASE_FRAGMENT), len(VALID_AMINO_ACIDS))
    assert result.logits.dtype == np.float32
    for aa in set(PETASE_FRAGMENT):
        assert aa in STUB_VOCAB
    assert sorted(STUB_VOCAB.values()) == list(range(len(VALID_AMINO_ACIDS)))


def test_stub_rows_are_valid_log_distributions() -> None:
    """Each row exponentiates to a normalized probability distribution (no -inf)."""
    result = StubConnector("esmc-600m").masked_sequence_logits(PETASE_FRAGMENT)
    probs = np.exp(result.logits.astype(np.float64))
    assert np.all(probs.sum(axis=1) == pytest.approx(1.0))
    assert np.all(result.logits > -np.inf)
