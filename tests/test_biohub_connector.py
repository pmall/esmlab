"""Covers the Biohub backend's guards, which hold without reaching the Platform."""

from typing import Any, cast

import pytest

from esmlab.connectors.biohub import MAX_MASKED_RESIDUES, BiohubConnector

SEQUENCE = "ACDEFGHIKLMNPQRSTVWY" * 3


def _connector() -> BiohubConnector:
    """A connector with no SDK client behind it.

    The guards under test run before anything touches ``self._client``, so the
    client is left unset rather than built: constructing the real one pulls in
    the whole ``esm`` stack for no benefit, and a test that needs a client
    supplies its own stand-in.
    """
    return object.__new__(BiohubConnector)


def test_a_region_over_the_mask_limit_is_refused() -> None:
    """The refusal happens before any request, which is the point of the limit.

    Every mask is a separate full-length request, so an over-long region has
    already cost its whole quadratic bill by the time the first answer lands.
    """
    with pytest.raises(ValueError, match="over the Biohub Platform limit"):
        _connector().masked_sequence_logits(SEQUENCE, 1, MAX_MASKED_RESIDUES + 1)


class _RefusingClient:
    """Stands in for the SDK client so the test never reaches the network."""

    def encode(self, protein):
        raise _Reached


class _Reached(Exception):
    """Raised by :class:`_RefusingClient` to prove a request was attempted."""


def test_the_limit_itself_is_allowed_through() -> None:
    """Exactly ``MAX_MASKED_RESIDUES`` masks passes the guard and dispatches.

    The stand-in client fails every request, so what this asserts is that the
    guard let the call through to the point of making them - the Platform is
    not under test here.
    """
    connector = _connector()
    # The stand-in is deliberately not an SDK client; only the guard is under test.
    cast(Any, connector)._client = _RefusingClient()

    with pytest.raises(RuntimeError, match="requests failed") as refused:
        connector.masked_sequence_logits(SEQUENCE, 1, MAX_MASKED_RESIDUES)

    assert f"{MAX_MASKED_RESIDUES}/{MAX_MASKED_RESIDUES}" in str(refused.value)


def test_the_region_check_still_runs_first() -> None:
    """Coordinates outside the sequence fail as coordinates, not as a mask count."""
    with pytest.raises(ValueError, match="outside 1-"):
        _connector().masked_sequence_logits(SEQUENCE, 1, len(SEQUENCE) + 100)
