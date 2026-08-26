from pathlib import Path

import numpy as np
import pytest

from esmlab.connectors.stub import STUB_VOCAB
from esmlab.seqio import (
    load_inference,
    parse_sequences,
    save_inference,
    validate_sequence,
)
from tests.fixtures import make_result


def test_validate_sequence_normalizes_and_rejects() -> None:
    assert validate_sequence(" acde ") == "ACDE"
    with pytest.raises(ValueError, match="non-canonical"):
        validate_sequence("ACXZ")
    with pytest.raises(ValueError, match="Empty"):
        validate_sequence("   ")


def test_parse_sequences_names_positional_inputs_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "input.fa"
    fasta.write_text(">my protein|1\nACDE\nFGHIK\n\n>dup-name\nMMMMM\n")
    fasta_duplicate = tmp_path / "other.fa"
    fasta_duplicate.write_text(">dup-name\nLLLLL\n")

    named = parse_sequences(["acdefghikl"], [fasta])
    assert [record.name for record in named] == ["seq_01", "my_protein_1", "dup-name"]
    assert named[0].sequence == "ACDEFGHIKL"

    with pytest.raises(ValueError, match="Duplicate sequence names"):
        parse_sequences([], [fasta, fasta_duplicate])
    with pytest.raises(ValueError, match="No input sequences"):
        parse_sequences([], [])


def test_inference_round_trip_preserves_everything(tmp_path: Path) -> None:
    result = make_result(
        "ACDE", [[0.0] * 20, [1.0] * 20, [0.5] * 20, [2.0] + [0.0] * 19]
    )
    run_dir = save_inference(
        tmp_path / "run_a", result, backend="stub", model="esmc-600m"
    )
    loaded, meta = load_inference(run_dir)

    assert loaded.sequence == "ACDE"
    np.testing.assert_array_equal(loaded.logits, result.logits)
    assert loaded.vocab == dict(STUB_VOCAB)
    assert meta.backend == "stub"
    assert meta.model == "esmc-600m"
    assert meta.name == "run_a"
    assert meta.created_utc


def test_save_inference_refuses_to_overwrite_existing_run(tmp_path: Path) -> None:
    result = make_result("A", [[0.0] * 20])
    save_inference(tmp_path / "run", result, backend="stub", model="esmc-300m")
    with pytest.raises(FileExistsError):
        save_inference(tmp_path / "run", result, backend="stub", model="esmc-300m")
