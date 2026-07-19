"""Shared fixtures for the test suite."""

import pytest

from src.storage.db import RatesStore


class FakeResponse:
    """Stand-in for requests.Response: canned .json() payload, optional
    exception raised from raise_for_status()."""

    def __init__(self, payload=None, status_error=None):
        self._payload = payload
        self._status_error = status_error

    def raise_for_status(self):
        if self._status_error is not None:
            raise self._status_error

    def json(self):
        return self._payload


@pytest.fixture
def fake_response_cls():
    """The FakeResponse class itself, so tests construct their own instances
    without importing from conftest (robust across pytest import modes)."""
    return FakeResponse


@pytest.fixture
def store(tmp_path):
    return RatesStore(tmp_path / "rates.csv")
