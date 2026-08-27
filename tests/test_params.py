from pathlib import Path

import pytest

from esmlab.connectors.base import ParamSpec, resolve_params

_DEVICE = ParamSpec(
    flag="--device",
    dest="device",
    env="",
    help="",
    choices=("auto", "cpu", "cuda"),
    default="auto",
)
_KEY = ParamSpec(
    flag="--key",
    dest="key",
    env="KEY",
    help="",
    required=True,
)
_INT = ParamSpec(
    flag="--n",
    dest="n",
    env="N",
    help="",
    type=int,
    default=10,
)
_PATH = ParamSpec(
    flag="--root",
    dest="root",
    env="",
    help="",
    type=Path,
    required=True,
)


def test_empty_specs_with_all_none_cli() -> None:
    assert resolve_params((), {"device": None}, label="backend 'stub'") == {}


def test_optional_default_applied() -> None:
    resolved = resolve_params((_DEVICE,), {"device": None}, label="x")
    assert resolved == {"device": "auto"}


def test_explicit_cli_value_overrides_default() -> None:
    resolved = resolve_params((_DEVICE,), {"device": "cpu"}, label="x")
    assert resolved == {"device": "cpu"}


def test_required_from_cli() -> None:
    resolved = resolve_params((_KEY,), {"key": "abc"}, label="x")
    assert resolved == {"key": "abc"}


def test_required_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEY", "from-env")
    resolved = resolve_params((_KEY,), {"key": None}, label="x")
    assert resolved == {"key": "from-env"}


def test_required_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KEY", raising=False)
    with pytest.raises(ValueError, match="required"):
        resolve_params((_KEY,), {"key": None}, label="x")


def test_irrelevant_param_raises() -> None:
    with pytest.raises(ValueError, match="does not apply"):
        resolve_params((_KEY,), {"key": "abc", "device": "cpu"}, label="x")


def test_cli_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEY", "from-env")
    resolved = resolve_params((_KEY,), {"key": "from-cli"}, label="x")
    assert resolved == {"key": "from-cli"}


def test_int_type_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("N", "42")
    resolved = resolve_params((_INT,), {"n": None}, label="x")
    assert resolved == {"n": 42}


def test_path_type_from_cli() -> None:
    resolved = resolve_params((_PATH,), {"root": Path("/tmp/cache")}, label="x")
    assert resolved == {"root": Path("/tmp/cache")}


def test_path_required_missing_raises() -> None:
    with pytest.raises(ValueError, match="required"):
        resolve_params((_PATH,), {"root": None}, label="x")
