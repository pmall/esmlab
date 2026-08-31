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

# Canonical model identifiers, grouped by the task family they serve. Each
# backend maps these to its own naming scheme (Biohub Platform names, HF repo
# ids). Sequence models answer masked-logits queries (ESMC); structure models
# predict 3D structure (ESMFold2, and its single-sequence "fast" variant).
CANONICAL_SEQUENCE_MODELS = ("esmc-300m", "esmc-600m", "esmc-6b")
CANONICAL_STRUCTURE_MODELS = ("esmfold2", "esmfold2-fast")


@dataclass(frozen=True)
class ParamSpec:
    """One CLI/env parameter, co-located with the backend it belongs to."""

    flag: str
    dest: str
    env: str
    help: str
    type: Callable[[str], str | int | Path] = str
    choices: tuple[str, ...] | None = None
    required: bool = False
    default: str | int | None = None


def load_env() -> None:
    """Loads variables from a .env file into os.environ if present.

    Called once by the CLI entrypoint before :func:`resolve_params` runs so
    that ``ParamSpec.env`` fallbacks (e.g. ``BIOHUB_API_KEY``) resolve.
    """
    from dotenv import load_dotenv

    load_dotenv()


def resolve_params(
    params: tuple[ParamSpec, ...],
    cli_values: Mapping[str, str | int | Path | None],
    *,
    label: str,
) -> dict[str, str | int | Path]:
    """Validates and resolves parameters for one backend.

    Irrelevant CLI values (non-None but not in ``params``) raise. Each spec is
    resolved from the CLI value, then its env var, then its default; required
    specs with no value raise.

    Called once per run from the CLI: the script passes the matching ``PARAMS``
    tuple (declared in each connector module) alongside the raw namespace
    values so backends stay self-contained.
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
    """Backend contract: mask each residue, read the logits, plus memory reporting.

    Every backend (stub, local, biohub, modal) implements
    :meth:`masked_sequence_logits`. Connectors only compute: persisting the
    result is the caller's job, via :mod:`esmlab.storage`.
    :meth:`peak_memory_bytes` exposes the peak memory of the last
    call where the backend can observe it (CUDA on the local backend); it
    returns ``None`` for backends with no observable memory (stub, biohub, modal,
    or the local backend on CPU).
    """

    def masked_sequence_logits(self, sequence: str) -> SequenceLogits:
        """Returns per-position masked-logits for ``sequence`` (axis 0 aligns to residues)."""
        ...

    def peak_memory_bytes(self) -> int | None:
        """Peak memory of the last :meth:`masked_sequence_logits` call, or ``None``.

        ``None`` means the backend cannot observe memory (stub, biohub, modal,
        or the local backend on CPU). The local backend on CUDA returns
        ``torch.cuda.max_memory_allocated`` reset around each call.
        """
        ...
