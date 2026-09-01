"""CLI/env parameter specs shared by every configurable layer.

Connectors and storage backends each co-locate a ``PARAMS`` tuple of
:class:`ParamSpec` objects describing their own CLI flags, env-var fallbacks,
and defaults. :func:`resolve_params` is generic over any such tuple, so adding
a parameter to a backend only touches that backend's module.
"""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

type ParamValue = str | int | Path


@dataclass(frozen=True)
class ParamSpec:
    """One CLI/env parameter, co-located with the layer it belongs to."""

    flag: str
    dest: str
    env: str
    help: str
    type: Callable[[str], ParamValue] = str
    choices: tuple[str, ...] | None = None
    required: bool = False
    default: ParamValue | None = None


def load_env() -> None:
    """Loads variables from a .env file into os.environ if present.

    Called once by the CLI entrypoint before :func:`resolve_params` runs so
    that ``ParamSpec.env`` fallbacks (e.g. ``BIOHUB_API_KEY``) resolve.
    """
    from dotenv import load_dotenv

    load_dotenv()


def resolve_params(
    params: tuple[ParamSpec, ...],
    cli_values: Mapping[str, ParamValue | None],
    *,
    label: str,
) -> dict[str, ParamValue]:
    """Validates and resolves parameters for one backend.

    Irrelevant CLI values (non-None but not in ``params``) raise. Each spec is
    resolved from the CLI value, then its env var, then its default; required
    specs with no value raise.

    Called once per run from the CLI: the script passes the matching ``PARAMS``
    tuple (declared in each connector or storage module) alongside the raw
    namespace values so backends stay self-contained.
    """
    spec_dests = {spec.dest for spec in params}
    for dest, value in cli_values.items():
        if value is not None and dest not in spec_dests:
            flag = f"--{dest.replace('_', '-')}"
            raise ValueError(f"{flag} does not apply to {label}")

    resolved: dict[str, ParamValue] = {}
    for spec in params:
        cli_value = cli_values.get(spec.dest)
        if cli_value is not None:
            resolved[spec.dest] = cli_value
            continue
        if spec.env:
            env_value = os.environ.get(spec.env)
            if env_value is not None:
                resolved[spec.dest] = spec.type(env_value)
                continue
        if spec.default is not None:
            resolved[spec.dest] = spec.default
            continue
        if spec.required:
            if spec.env:
                raise ValueError(
                    f"{spec.flag} (or ${spec.env}) is required for {label}"
                )
            raise ValueError(f"{spec.flag} is required for {label}")
    return resolved
