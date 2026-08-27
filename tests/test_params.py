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
    """A backend with no params (e.g. stub) resolves to an empty dict."""
    assert resolve_params((), {"device": None}, label="backend 'stub'") == {}


def test_optional_default_applied() -> None:
    """An optional spec with no CLI value falls back to its default."""
    resolved = resolve_params((_DEVICE,), {"device": None}, label="x")
    assert resolved == {"device": "auto"}


def test_explicit_cli_value_overrides_default() -> None:
    """A provided CLI value wins over the spec default."""
    resolved = resolve_params((_DEVICE,), {"device": "cpu"}, label="x")
    assert resolved == {"device": "cpu"}


def test_required_from_cli() -> None:
    """A required spec is satisfied by a CLI value."""
    resolved = resolve_params((_KEY,), {"key": "abc"}, label="x")
    assert resolved == {"key": "abc"}


def test_required_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A required spec with no CLI value is satisfied by its env var."""
    monkeypatch.setenv("KEY", "from-env")
    resolved = resolve_params((_KEY,), {"key": None}, label="x")
    assert resolved == {"key": "from-env"}


def test_required_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A required spec with neither CLI, env, nor default raises."""
    monkeypatch.delenv("KEY", raising=False)
    with pytest.raises(ValueError, match="required"):
        resolve_params((_KEY,), {"key": None}, label="x")


def test_irrelevant_param_raises() -> None:
    """A non-None CLI value for a spec the backend doesn't have is rejected."""
    with pytest.raises(ValueError, match="does not apply"):
        resolve_params((_KEY,), {"key": "abc", "device": "cpu"}, label="x")


def test_cli_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI values take precedence over env-var fallbacks."""
    monkeypatch.setenv("KEY", "from-env")
    resolved = resolve_params((_KEY,), {"key": "from-cli"}, label="x")
    assert resolved == {"key": "from-cli"}


def test_int_type_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env values are coerced through the spec's ``type`` callable."""
    monkeypatch.setenv("N", "42")
    resolved = resolve_params((_INT,), {"n": None}, label="x")
    assert resolved == {"n": 42}


def test_path_type_from_cli() -> None:
    """A Path-typed spec keeps a Path value from the CLI."""
    resolved = resolve_params((_PATH,), {"root": Path("/tmp/cache")}, label="x")
    assert resolved == {"root": Path("/tmp/cache")}


def test_path_required_missing_raises() -> None:
    """A required Path spec with no source raises."""
    with pytest.raises(ValueError, match="required"):
        resolve_params((_PATH,), {"root": None}, label="x")
