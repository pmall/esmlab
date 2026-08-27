"""Connector contract and shared parameter-spec infrastructure.

Each backend module co-locates a ``PARAMS`` tuple of :class:`ParamSpec`
objects describing its own CLI flags, env-var fallbacks, and defaults.
:func:`resolve_params` is generic over any such tuple, so adding a parameter
to a backend only touches that backend's module.
"""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

# Canonical model identifiers accepted by every backend factory; each backend
# maps these to its own naming scheme (Forge names, HF repo ids).
CANONICAL_MODELS = ("esmc-300m", "esmc-600m")


@dataclass(frozen=True)
class ParamSpec:
    """One CLI/env parameter, co-located with the backend or cache it belongs to."""

    flag: str
    dest: str
    env: str
    help: str
    type: Callable[[str], str | int | Path] = str
    choices: tuple[str, ...] | None = None
    required: bool = False
    default: str | int | None = None


def load_env() -> None:
    """Loads variables from a .env file into os.environ if present."""
    from dotenv import load_dotenv

    load_dotenv()


def resolve_params(
    params: tuple[ParamSpec, ...],
    cli_values: Mapping[str, str | int | Path | None],
    *,
    label: str,
) -> dict[str, str | int | Path]:
    """Validates and resolves parameters for one backend or cache.

    Irrelevant CLI values (non-None but not in ``params``) raise. Each spec is
    resolved from the CLI value, then its env var, then its default; required
    specs with no value raise.
    """
    spec_dests = {spec.dest for spec in params}
    for dest, value in cli_values.items():
        if value is not None and dest not in spec_dests:
            flag = f"--{dest.replace('_', '-')}"
            raise ValueError(f"{flag} does not apply to {label}")

    resolved: dict[str, str | int | Path] = {}
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


@dataclass(frozen=True)
class SequenceLogits:
    """Masked-language-model logits for one sequence.

    ``logits[i]`` holds the token-distribution row predicted when residue i of
    ``sequence`` was the masked position; BOS/EOS rows are stripped so axis 0
    maps one-to-one onto sequence residues. ``vocab`` maps token strings to
    column indices of the logits array.
    """

    sequence: str
    logits: npt.NDArray[np.float32]
    vocab: Mapping[str, int]


class ModelConnector(Protocol):
    """One method is all analysis needs: mask each residue, read the logits."""

    def masked_sequence_logits(self, sequence: str) -> SequenceLogits: ...
