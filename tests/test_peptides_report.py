"""Covers the report stage: one page per stored entry, the index, free re-runs."""

import json
import re
from pathlib import Path

import pytest

from esmlab.amino_acids import VALID_AMINO_ACIDS
from esmlab.connectors.stub import StubConnector
from esmlab.inference import InferenceSettings, run_inference
from esmlab.peptides_report import ReportSettings, run_report
from esmlab.seqio import NamedSequence
from esmlab.storage import StorageSettings, logits_key, open_storage
from tests.fixtures import named, sqlite_settings, whole

SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"
OTHER_SEQUENCE = "MKTAYIAKQRQISFVK"


def _populate(
    tmp_path: Path,
    sequences: list[NamedSequence],
    *,
    model: str = "esmc-600m",
) -> StorageSettings:
    """Runs the stub compute stage so the report stage has entries to read."""
    storage = sqlite_settings(tmp_path / "logits.sqlite3")
    run_inference(
        InferenceSettings(
            backend="stub",
            model=model,
            device="",
            batch_size=0,
            biohub_api_key="",
            modal_token_id="",
            modal_token_secret="",
            modal_gpu="",
            sequences=sequences,
            storage=storage,
            perf_report=tmp_path / "performance.csv",
        )
    )
    return storage


def _settings(
    tmp_path: Path,
    *,
    threshold: float = 0.8,
    backend: str | None = "stub",
    model: str | None = "esmc-600m",
) -> ReportSettings:
    return ReportSettings(
        backend=backend,
        model=model,
        storage=sqlite_settings(tmp_path / "logits.sqlite3"),
        out_dir=tmp_path / "reports",
        threshold=threshold,
        top_k=10,
    )


def _payload(page: Path) -> dict:
    """Reads back the JSON island a rendered page carries."""
    island = re.search(
        r'<script type="application/json" id="payload">(.*?)</script>',
        page.read_text(),
        re.DOTALL,
    )
    assert island is not None
    return json.loads(island.group(1))


def _page(
    tmp_path: Path,
    sequence: str = SEQUENCE,
    backend: str = "stub",
    model: str = "esmc-600m",
    start: int = 1,
    stop: int | None = None,
) -> Path:
    """Where the report stage writes one request's page."""
    key = logits_key(sequence, start, len(sequence) if stop is None else stop)
    return tmp_path / "reports" / "peptides" / backend / model / f"{key}.html"


def _store_under(
    storage: StorageSettings, backend: str, sequences: list[NamedSequence]
) -> None:
    """Files stub logits under another backend's name, without running that backend.

    Grouping entries by what produced them is what the report path now
    expresses, and a second backend's rows are what proves it; no test may run
    a backend that costs money or downloads a checkpoint.
    """
    store = open_storage(storage)
    connector = StubConnector("esmc-600m")
    for record in sequences:
        store.save(
            connector.masked_sequence_logits(
                record.sequence, record.start, record.stop
            ),
            backend=backend,
            model="esmc-600m",
            label=record.name,
            metadata={},
        )


def test_a_run_writes_one_page_per_entry_and_an_index(tmp_path: Path) -> None:
    """The page is the whole report: no sidecar files accompany it."""
    _populate(tmp_path, [named("tiny", SEQUENCE)])
    settings = _settings(tmp_path)

    run_report(settings)

    assert {
        path.name
        for path in (settings.out_dir / "peptides" / "stub" / "esmc-600m").iterdir()
    } == {
        f"{logits_key(SEQUENCE, 1, len(SEQUENCE))}.html",
        "index.html",
    }


def test_entry_page_carries_its_whole_payload(tmp_path: Path) -> None:
    """The page is self-contained: the label and every array live in its JSON island.

    Guards the property the whole design rests on - the template is static and
    Python only supplies data - so a page can be opened straight from disk.
    """
    _populate(tmp_path, [named("tiny", SEQUENCE)])
    settings = _settings(tmp_path)

    run_report(settings)

    payload = _payload(_page(tmp_path))
    assert payload["label"] == "tiny"
    assert payload["sequence"] == SEQUENCE
    assert len(payload["entropy"]) == len(SEQUENCE)
    assert len(payload["llr"]) == len(SEQUENCE)
    assert len(payload["llr"][0]) == len(VALID_AMINO_ACIDS)
    assert sorted(payload["amino_acids"]) == sorted(VALID_AMINO_ACIDS)


def test_index_maps_keys_back_to_labels(tmp_path: Path) -> None:
    """The label survives only as index data, never as a path component."""
    _populate(tmp_path, [named("tiny", SEQUENCE)])
    settings = _settings(tmp_path)

    run_report(settings)

    payload = _payload(
        settings.out_dir / "peptides" / "stub" / "esmc-600m" / "index.html"
    )
    assert (payload["backend"], payload["model"]) == ("stub", "esmc-600m")
    assert payload["entries"] == [
        {
            "key": logits_key(SEQUENCE, 1, len(SEQUENCE)),
            "label": "tiny",
            "length": len(SEQUENCE),
            "start": 1,
            "stop": len(SEQUENCE),
            "full_length": len(SEQUENCE),
            "created_utc": payload["entries"][0]["created_utc"],
        }
    ]


def test_a_region_scores_and_reports_only_the_peptide(tmp_path: Path) -> None:
    """A region is a compute scope: only its residues are masked and reported.

    This is the point of naming coordinates - masking a 15-residue peptide
    inside a 500-residue protein should cost 15 forward passes, not 500 - and
    the page is about the peptide, numbered 1..n.
    """
    _populate(tmp_path, [named("peptide", SEQUENCE, 5, 9)])
    settings = _settings(tmp_path)

    run_report(settings)

    payload = _payload(_page(tmp_path, start=5, stop=9))
    assert (payload["start"], payload["stop"]) == (5, 9)
    assert payload["full_sequence"] == SEQUENCE
    assert payload["sequence"] == SEQUENCE[4:9]
    assert payload["positions"] == [1, 2, 3, 4, 5]
    assert len(payload["entropy"]) == 5
    assert len(payload["llr"]) == 5
    assert all(1 <= row["position"] <= 5 for row in payload["most_constrained"])
    assert all(1 <= row["position"] <= 5 for row in payload["top_substitutions"])


def test_a_region_stores_only_its_own_rows_against_the_whole_sequence(
    tmp_path: Path,
) -> None:
    """The model saw the whole protein; only the region's positions were masked.

    Storing the full sequence is what keeps the context recoverable - the rows
    mean nothing without it - while the row count is the saving.
    """
    storage = open_storage(_populate(tmp_path, [named("peptide", SEQUENCE, 5, 9)]))

    stored = storage.load(
        backend="stub", model="esmc-600m", sequence=SEQUENCE, start=5, stop=9
    )

    assert stored.logits.sequence == SEQUENCE
    assert (stored.logits.start, stored.logits.stop) == (5, 9)
    assert stored.logits.logits.shape[0] == 5
    assert stored.logits.residues == SEQUENCE[4:9]


def test_rerun_with_new_threshold_touches_no_storage(tmp_path: Path) -> None:
    """Re-rendering is pure CPU work: the tolerant column flips, storage is untouched.

    This is the payoff of splitting compute from reporting - the old unified
    pipeline re-ran the model to change one presentation knob.
    """
    storage = open_storage(_populate(tmp_path, [named("tiny", SEQUENCE)]))
    before = storage.load(
        backend="stub", model="esmc-600m", sequence=SEQUENCE, **whole(SEQUENCE)
    )

    run_report(_settings(tmp_path, threshold=0.0))
    strict = sum(_payload(_page(tmp_path))["tolerant"])

    run_report(_settings(tmp_path, threshold=1.0))
    loose = sum(_payload(_page(tmp_path))["tolerant"])

    assert loose > strict
    # A rewritten row would carry a fresh created_utc.
    after = storage.load(
        backend="stub", model="esmc-600m", sequence=SEQUENCE, **whole(SEQUENCE)
    )
    assert after.created_utc == before.created_utc


def test_two_records_sharing_a_label_both_get_reports(tmp_path: Path) -> None:
    """Duplicate FASTA headers are fine now that keys come from the sequence."""
    _populate(
        tmp_path,
        [named("dup", SEQUENCE), named("dup", OTHER_SEQUENCE)],
    )
    settings = _settings(tmp_path)

    run_report(settings)

    model_dir = settings.out_dir / "peptides" / "stub" / "esmc-600m"
    assert (model_dir / f"{logits_key(SEQUENCE, 1, len(SEQUENCE))}.html").is_file()
    assert (
        model_dir / f"{logits_key(OTHER_SEQUENCE, 1, len(OTHER_SEQUENCE))}.html"
    ).is_file()


def test_no_model_reports_every_model_in_one_run(tmp_path: Path) -> None:
    """The default renders the whole store, each model with its own index.

    The model is in the report path, so the same sequence scored twice does not
    overwrite itself.
    """
    _populate(tmp_path, [named("tiny", SEQUENCE)], model="esmc-600m")
    _populate(tmp_path, [named("tiny", SEQUENCE)], model="esmc-300m")

    run_report(_settings(tmp_path, model=None))

    for model in ("esmc-600m", "esmc-300m"):
        assert _page(tmp_path, model=model).is_file()
        index = _payload(
            tmp_path / "reports" / "peptides" / "stub" / model / "index.html"
        )
        assert index["model"] == model
        assert [row["key"] for row in index["entries"]] == [
            logits_key(SEQUENCE, 1, len(SEQUENCE))
        ]


def test_a_named_model_reports_only_that_model(tmp_path: Path) -> None:
    """``--model`` narrows the run: the other model's pages are not written."""
    _populate(tmp_path, [named("tiny", SEQUENCE)], model="esmc-600m")
    _populate(tmp_path, [named("tiny", SEQUENCE)], model="esmc-300m")

    run_report(_settings(tmp_path, model="esmc-300m"))

    assert _page(tmp_path, model="esmc-300m").is_file()
    assert not (tmp_path / "reports" / "peptides" / "stub" / "esmc-600m").exists()


def test_each_backend_gets_its_own_directory(tmp_path: Path) -> None:
    """One sequence scored by two backends is two pages, not one overwritten.

    The storage key is the sequence's alone, so what produced the entry has to
    be in the path for both results to survive.
    """
    storage = _populate(tmp_path, [named("tiny", SEQUENCE)])
    _store_under(storage, "local", [named("tiny", SEQUENCE)])

    run_report(_settings(tmp_path, backend=None))

    assert _page(tmp_path, backend="stub").is_file()
    assert _page(tmp_path, backend="local").is_file()


def test_a_named_backend_reports_only_that_backend(tmp_path: Path) -> None:
    """``--backend`` narrows the run the same way ``--model`` does."""
    storage = _populate(tmp_path, [named("tiny", SEQUENCE)])
    _store_under(storage, "local", [named("tiny", SEQUENCE)])

    run_report(_settings(tmp_path, backend="local"))

    assert _page(tmp_path, backend="local").is_file()
    assert not (tmp_path / "reports" / "peptides" / "stub").exists()


def test_reporting_a_backend_with_no_entries_names_the_stored_ones(
    tmp_path: Path,
) -> None:
    """An unstored backend fails like an unstored model, naming the pairs held."""
    _populate(tmp_path, [named("tiny", SEQUENCE)])
    settings = _settings(tmp_path, backend="modal")

    with pytest.raises(ValueError, match=r"stub/esmc-600m \(1\)"):
        run_report(settings)

    assert not settings.out_dir.exists()


def test_reporting_a_model_with_no_entries_names_the_stored_ones(
    tmp_path: Path,
) -> None:
    """An unstored model is a CLI error, not an empty report directory.

    Naming a model with no entries almost always means the compute run used a
    different one; the message has to say which, and nothing may be written
    under the wrong name.
    """
    _populate(tmp_path, [named("tiny", SEQUENCE)], model="esmc-300m")
    settings = _settings(tmp_path, model="esmc-600m")

    with pytest.raises(ValueError, match=r"esmc-300m \(1\)"):
        run_report(settings)

    assert not settings.out_dir.exists()


def test_reporting_an_empty_storage_is_an_error(tmp_path: Path) -> None:
    """Same for a store that holds nothing at all."""
    settings = _settings(tmp_path)

    with pytest.raises(ValueError, match="no entries at all"):
        run_report(settings)
