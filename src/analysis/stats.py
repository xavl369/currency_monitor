"""Descriptive statistics for the USD/MXN series: log returns, rolling
mean/volatility, and a backward-looking "last known value" display helper.

The series keeps its natural weekend/holiday gaps — pandas handles a datetime
index with gaps fine, and forward-filling would falsely imply "no change" on
days that never traded. `get_last_known_value` exists for the display layer
(dashboard/email) to answer "what's the dollar worth today" on a Saturday;
its result is never written back into data/rates.csv.

Standalone usage (quick console summary of data/rates.csv):
    python -m src.analysis.stats
"""

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.storage.db import RatesStore

ROOT = Path(__file__).resolve().parent.parent.parent
RATES_CSV = ROOT / "data" / "rates.csv"

TRADING_DAYS_PER_YEAR = 252


class RateStats:
    """Descriptive/rolling statistics over a USD/MXN rate series
    (float values on a sorted DatetimeIndex with natural gaps)."""

    def __init__(self, series: pd.Series):
        self.series = series

    @classmethod
    def from_store(cls, store: RatesStore) -> "RateStats":
        """Build from a RatesStore, converting its (date_iso, rate_str) rows."""
        rows = store.read()
        series = pd.Series(
            [float(rate) for _, rate in rows],
            index=pd.DatetimeIndex([d for d, _ in rows], name="date"),
            name="rate",
        ).sort_index()
        return cls(series)

    def log_returns(self) -> pd.Series:
        """Day-over-day log returns between consecutive observations."""
        return np.log(self.series).diff().dropna()

    def rolling_mean(self, window: int = 30) -> pd.Series:
        """Rolling mean of the rate over the last `window` observations."""
        return self.series.rolling(window).mean().dropna()

    def rolling_volatility(self, window: int = 21, annualize: bool = True) -> pd.Series:
        """Rolling std of log returns over `window` observations, annualized
        by sqrt(252) unless annualize=False."""
        vol = self.log_returns().rolling(window).std()
        if annualize:
            vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
        return vol.dropna()

    def descriptive(self) -> dict:
        """Summary stats for the rate levels and their log returns."""
        s = self.series
        r = self.log_returns()
        return {
            "levels": {
                "observations": int(s.size),
                "start": s.index[0].strftime("%Y-%m-%d"),
                "end": s.index[-1].strftime("%Y-%m-%d"),
                "latest": float(s.iloc[-1]),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "min_date": s.idxmin().strftime("%Y-%m-%d"),
                "max": float(s.max()),
                "max_date": s.idxmax().strftime("%Y-%m-%d"),
            },
            "log_returns": {
                "mean": float(r.mean()),
                "std": float(r.std()),
                "annualized_vol": float(r.std() * np.sqrt(TRADING_DAYS_PER_YEAR)),
                "skew": float(r.skew()),
                "kurtosis": float(r.kurtosis()),
                "worst_day": float(r.min()),
                "worst_day_date": r.idxmin().strftime("%Y-%m-%d"),
                "best_day": float(r.max()),
                "best_day_date": r.idxmax().strftime("%Y-%m-%d"),
            },
        }

    def get_last_known_value(self, when: str | date | datetime) -> tuple[str, float] | None:
        """Most recent (date_iso, rate) on or before `when` — for any date,
        including weekends/holidays. Display-only: never write this back into
        the dataset. Returns None if `when` predates the whole series."""
        upto = self.series.loc[: pd.Timestamp(when)]
        if upto.empty:
            return None
        return upto.index[-1].strftime("%Y-%m-%d"), float(upto.iloc[-1])


def main() -> int:
    stats = RateStats.from_store(RatesStore(RATES_CSV))

    d = stats.descriptive()
    lv, lr = d["levels"], d["log_returns"]
    print(f"USD/MXN — {lv['observations']} observations, {lv['start']} to {lv['end']}")
    print(f"  latest: {lv['latest']:.4f}   mean: {lv['mean']:.4f}   std: {lv['std']:.4f}")
    print(f"  min: {lv['min']:.4f} ({lv['min_date']})   max: {lv['max']:.4f} ({lv['max_date']})")
    print("Log returns:")
    print(f"  mean: {lr['mean']:+.6f}   daily std: {lr['std']:.6f}   annualized vol: {lr['annualized_vol']:.2%}")
    print(f"  skew: {lr['skew']:.2f}   excess kurtosis: {lr['kurtosis']:.2f}")
    print(f"  worst day: {lr['worst_day']:+.2%} ({lr['worst_day_date']})   best day: {lr['best_day']:+.2%} ({lr['best_day_date']})")

    vol = stats.rolling_volatility()
    print(f"Rolling 21-day annualized vol (latest): {vol.iloc[-1]:.2%}")

    today = date.today().isoformat()
    known = stats.get_last_known_value(today)
    if known is not None:
        print(f"Last known value as of {today}: {known[1]:.4f} (from {known[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
