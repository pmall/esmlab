"""Covers the filesystem logits storage: keys, layout, round-trip, atomicity."""

import json
from pathlib import Path

import numpy as np
import pytest

from esmlab.connectors.stub import StubConnector
from esmlab.storage import (
    LOGITS_FILE,
    META_FILE,
    FileLogitsStorage,
    logits_key,
)

SEQUENCE = "MKTAYIAKQRQISFVK"
OTHER_SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQ"


def _result(sequence: str = SEQUENCE):
    """Deterministic logits for a sequence, without touching a real model."""
    return StubConnector("esmc-600m").masked_sequence_logits(sequence)


def test_missing_entry_is_absent_and_load_raises(tmp_path: Path) -> None:
    """An unwritten entry reports absent and refuses to load."""
    storage = FileLogitsStorage(tmp_path)

    assert storage.has(model="esmc-600m", sequence=SEQUENCE) is False
    with pytest.raises(KeyError):
        storage.load(model="esmc-600m", sequence=SEQUENCE)


def test_save_then_load_round_trips_result_and_provenance(tmp_path: Path) -> None:
    """A saved entry comes back with identical logits plus its display label."""
    storage = FileLogitsStorage(tmp_path)
    result = _result()

    storage.save(result, model="esmc-600m", label="stat1")

    assert storage.has(model="esmc-600m", sequence=SEQUENCE) is True
    stored = storage.load(model="esmc-600m", sequence=SEQUENCE)
    assert stored.model == "esmc-600m"
    assert stored.label == "stat1"
    assert stored.created_utc
    assert stored.logits.sequence == SEQUENCE
    assert stored.logits.vocab == dict(result.vocab)
    np.testing.assert_array_equal(stored.logits.logits, result.logits)


def test_key_depends_on_sequence_alone(tmp_path: Path) -> None:
    """The digest is the sequence's; the model selects the directory instead."""
    assert logits_key(SEQUENCE) == logits_key(SEQUENCE)
    assert logits_key(SEQUENCE) != logits_key(OTHER_SEQUENCE)

    storage = FileLogitsStorage(tmp_path)
    storage.save(_result(), model="esmc-600m", label="a")
    storage.save(_result(), model="esmc-300m", label="a")

    key = logits_key(SEQUENCE)
    assert (tmp_path / "esmc-600m" / key / META_FILE).is_file()
    assert (tmp_path / "esmc-300m" / key / META_FILE).is_file()
    # One sequence has one leaf name everywhere, so a glob finds every model.
    assert sorted(path.parent.name for path in tmp_path.glob(f"*/{key}/")) == [
        "esmc-300m",
        "esmc-600m",
    ]


def test_entry_for_a_different_sequence_is_separate(tmp_path: Path) -> None:
    """Distinct sequences never share an entry under the same model."""
    storage = FileLogitsStorage(tmp_path)
    storage.save(_result(), model="esmc-600m", label="a")

    assert storage.has(model="esmc-600m", sequence=OTHER_SEQUENCE) is False


def test_saving_twice_overwrites_in_place(tmp_path: Path) -> None:
    """A recompute lands on the same entry rather than fragmenting the store."""
    storage = FileLogitsStorage(tmp_path)
    storage.save(_result(), model="esmc-600m", label="first")
    storage.save(_result(), model="esmc-600m", label="second")

    assert len(list((tmp_path / "esmc-600m").iterdir())) == 1
    assert storage.load(model="esmc-600m", sequence=SEQUENCE).label == "second"


def test_torn_write_without_sidecar_reports_absent(tmp_path: Path) -> None:
    """Presence is the sidecar, so a half-written entry is not a hit.

    Locks in the atomic-write contract: persistence is unconditional, so a
    directory holding only the npz must be recomputed, not skipped forever.
    """
    entry_dir = tmp_path / "esmc-600m" / logits_key(SEQUENCE)
    entry_dir.mkdir(parents=True)
    np.savez_compressed(entry_dir / LOGITS_FILE, logits=_result().logits)

    assert (
        FileLogitsStorage(tmp_path).has(model="esmc-600m", sequence=SEQUENCE) is False
    )


def test_save_leaves_no_staging_directory(tmp_path: Path) -> None:
    """The temporary staging directory is cleaned up after a successful save."""
    FileLogitsStorage(tmp_path).save(_result(), model="esmc-600m", label="a")

    assert [path.name for path in (tmp_path / "esmc-600m").iterdir()] == [
        logits_key(SEQUENCE)
    ]


def test_meta_json_holds_provenance_and_no_identity_fields(tmp_path: Path) -> None:
    """The sidecar describes the entry; it carries no backend and no digest-as-name."""
    FileLogitsStorage(tmp_path).save(_result(), model="esmc-600m", label="stat1")

    payload = json.loads(
        (tmp_path / "esmc-600m" / logits_key(SEQUENCE) / META_FILE).read_text()
    )
    assert set(payload) == {"model", "label", "created_utc", "sequence", "vocab"}
    assert payload["model"] == "esmc-600m"
    assert payload["label"] == "stat1"
    assert payload["sequence"] == SEQUENCE


def test_entries_yields_only_the_requested_model(tmp_path: Path) -> None:
    """Enumeration is per-model and skips incomplete directories."""
    storage = FileLogitsStorage(tmp_path)
    storage.save(_result(), model="esmc-600m", label="a")
    storage.save(_result(OTHER_SEQUENCE), model="esmc-600m", label="b")
    storage.save(_result(), model="esmc-300m", label="c")

    labels = sorted(entry.label for entry in storage.entries(model="esmc-600m"))
    assert labels == ["a", "b"]
    assert [entry.label for entry in storage.entries(model="esmc-300m")] == ["c"]


def test_entries_on_unknown_model_is_empty(tmp_path: Path) -> None:
    """A model never written to yields nothing rather than raising."""
    assert list(FileLogitsStorage(tmp_path).entries(model="esmc-6b")) == []
