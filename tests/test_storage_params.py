"""Covers storage selection on the CLI: defaults, env fallbacks, cross-backend flags."""

import argparse
from pathlib import Path

import pytest

from esmlab.storage import add_storage_arguments, storage_settings

# Stands in for a script's own store: each one names the database its topic
# writes, and the CLI carries that default through to the resolved settings.
TOPIC_SQLITE_PATH = Path("data/peptides.sqlite")


def _parse(argv: list[str]) -> argparse.Namespace:
    """Parses ``argv`` with only the storage flags declared, as every script does."""
    parser = argparse.ArgumentParser()
    add_storage_arguments(parser, sqlite_default=TOPIC_SQLITE_PATH)
    return parser.parse_args(argv)


def test_sqlite_is_the_default_and_lands_on_the_scripts_own_database() -> None:
    """No flags at all resolves to the database the calling script named."""
    settings = storage_settings(_parse([]), sqlite_default=TOPIC_SQLITE_PATH)

    assert settings.storage == "sqlite"
    assert settings.sqlite_path == TOPIC_SQLITE_PATH


def test_sqlite_path_flag_overrides_the_default() -> None:
    """A stub run points at its own file so fake logits never reach a real run."""
    settings = storage_settings(
        _parse(["--sqlite-path", "data/logits-stub.sqlite3"]),
        sqlite_default=TOPIC_SQLITE_PATH,
    )

    assert settings.sqlite_path == Path("data/logits-stub.sqlite3")


def test_postgres_falls_back_to_env_and_defaults(monkeypatch) -> None:
    """Credentials come from the environment; host, port and dbname have defaults."""
    monkeypatch.setenv("POSTGRES_USER", "esm")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")

    settings = storage_settings(
        _parse(["--storage", "postgres"]), sqlite_default=TOPIC_SQLITE_PATH
    )

    assert settings.postgres_user == "esm"
    assert settings.postgres_password == "secret"
    assert settings.postgres_host == "localhost"
    assert settings.postgres_port == 5432
    assert settings.postgres_dbname == "esmlab"


def test_postgres_flags_beat_the_environment(monkeypatch) -> None:
    """An explicit flag wins over its env var, matching the backend params."""
    monkeypatch.setenv("POSTGRES_HOST", "from-env")
    monkeypatch.setenv("POSTGRES_USER", "esm")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")

    settings = storage_settings(
        _parse(["--storage", "postgres", "--postgres-host", "from-cli"]),
        sqlite_default=TOPIC_SQLITE_PATH,
    )

    assert settings.postgres_host == "from-cli"


def test_postgres_without_credentials_is_rejected(monkeypatch) -> None:
    """A required parameter with no flag and no env var fails before connecting."""
    monkeypatch.delenv("POSTGRES_USER", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)

    with pytest.raises(ValueError, match="--postgres-user"):
        storage_settings(
            _parse(["--storage", "postgres"]), sqlite_default=TOPIC_SQLITE_PATH
        )


def test_postgres_flag_on_a_sqlite_run_is_rejected() -> None:
    """Flags belonging to the other backend raise instead of being ignored."""
    with pytest.raises(ValueError, match="--postgres-host does not apply"):
        storage_settings(
            _parse(["--postgres-host", "db.internal"]), sqlite_default=TOPIC_SQLITE_PATH
        )


def test_sqlite_flag_on_a_postgres_run_is_rejected(monkeypatch) -> None:
    """The check runs both ways, so a stale --sqlite-path is never silently dropped."""
    monkeypatch.setenv("POSTGRES_USER", "esm")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")

    with pytest.raises(ValueError, match="--sqlite-path does not apply"):
        storage_settings(
            _parse(["--storage", "postgres", "--sqlite-path", "x.db"]),
            sqlite_default=TOPIC_SQLITE_PATH,
        )


def test_describe_names_the_resource_without_the_password(monkeypatch) -> None:
    """This string reaches stdout and the perf CSV, so it states the exact form."""
    monkeypatch.setenv("POSTGRES_USER", "esm")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
    settings = storage_settings(
        _parse(["--storage", "postgres"]), sqlite_default=TOPIC_SQLITE_PATH
    )

    assert settings.describe() == "postgres:esm@localhost:5432/esmlab"


def test_sqlite_flags_round_trip_into_the_report_command() -> None:
    """The command a compute script prints must reopen the same store."""
    settings = storage_settings(
        _parse(["--sqlite-path", "data/run.sqlite3"]), sqlite_default=TOPIC_SQLITE_PATH
    )

    assert (
        storage_settings(
            _parse(settings.flags().split()), sqlite_default=TOPIC_SQLITE_PATH
        )
        == settings
    )
