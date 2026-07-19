"""Orchestration tests for DailyCollector (src/collector/run_daily.py) using
injected fakes, plus the spec's literal "collector idempotency" check against
a real RatesStore in tmp_path.

Importing run_daily pulls in src.email.notifier — and matplotlib (headless
Agg backend) — at module top; that's expected, and why matplotlib is in
requirements-dev.txt.
"""

import pytest

from src.collector.run_daily import DailyCollector
from src.storage.db import RatesStore


class FakeStore:
    def __init__(self, added=True):
        self.added = added
        self.calls = []

    def append(self, date_iso, rate):
        self.calls.append((date_iso, rate))
        return self.added


class FakeBanxico:
    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc
        self.calls = 0

    def fetch_fix_rate(self):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.result


class FakeFallback:
    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc
        self.calls = 0

    def fetch_rate(self):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.result


def test_banxico_success_wins():
    banxico = FakeBanxico(result=("2026-07-17", "18.6543"))
    fallback = FakeFallback(result=("2026-07-17", "18.70"))
    collector = DailyCollector(FakeStore(), fallback, banxico)

    assert collector.fetch_latest() == ("2026-07-17", "18.6543", "Banxico")
    assert fallback.calls == 0


def test_banxico_none_goes_straight_to_fallback():
    fallback = FakeFallback(result=("2026-07-17", "18.70"))
    collector = DailyCollector(FakeStore(), fallback, banxico=None)

    assert collector.fetch_latest() == ("2026-07-17", "18.70", "fallback (Frankfurter)")
    assert fallback.calls == 1


def test_banxico_exception_falls_back(capsys):
    banxico = FakeBanxico(exc=RuntimeError("boom"))
    fallback = FakeFallback(result=("2026-07-17", "18.70"))
    collector = DailyCollector(FakeStore(), fallback, banxico)

    assert collector.fetch_latest() == ("2026-07-17", "18.70", "fallback (Frankfurter)")
    assert banxico.calls == 1
    assert fallback.calls == 1
    assert "Banxico fetch failed" in capsys.readouterr().err


def test_fallback_exception_propagates():
    collector = DailyCollector(
        FakeStore(), FakeFallback(exc=ConnectionError("down")), banxico=None
    )
    with pytest.raises(ConnectionError):
        collector.fetch_latest()


def test_both_sources_fail_raises_fallback_error():
    collector = DailyCollector(
        FakeStore(),
        FakeFallback(exc=ConnectionError("fallback down")),
        FakeBanxico(exc=RuntimeError("banxico down")),
    )
    with pytest.raises(ConnectionError, match="fallback down"):
        collector.fetch_latest()


def test_run_appends_fetched_row_and_returns_added_true():
    store = FakeStore(added=True)
    collector = DailyCollector(
        store, FakeFallback(), FakeBanxico(result=("2026-07-17", "18.6543"))
    )

    assert collector.run() == ("2026-07-17", "18.6543", "Banxico", True)
    assert store.calls == [("2026-07-17", "18.6543")]


def test_run_proxies_added_false_from_store():
    # False is the "already stored today, skip the email" signal.
    collector = DailyCollector(
        FakeStore(added=False), FakeFallback(), FakeBanxico(result=("2026-07-17", "18.65"))
    )
    assert collector.run()[3] is False


def test_collector_idempotent_with_real_store(tmp_path):
    store = RatesStore(tmp_path / "rates.csv")
    collector = DailyCollector(
        store, FakeFallback(), FakeBanxico(result=("2026-07-17", "18.6543"))
    )

    assert collector.run() == ("2026-07-17", "18.6543", "Banxico", True)
    first_bytes = store.path.read_bytes()

    assert collector.run() == ("2026-07-17", "18.6543", "Banxico", False)
    assert store.path.read_bytes() == first_bytes
    assert store.read() == [("2026-07-17", "18.6543")]
