"""Time-series structure of the USD/MXN series: STL decomposition and
ADF/KPSS stationarity tests.

The series is used as-is, with its natural weekend/holiday gaps — observations
are treated as consecutive trading days (standard for daily FX), never
forward-filled. STL therefore needs an explicit `period` in observations:
the default 252 approximates one trading year.

Standalone usage (quick console check against data/rates.csv):
    python -m src.analysis.decomposition
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.seasonal import STL, DecomposeResult
from statsmodels.tsa.stattools import adfuller, kpss

from src.analysis.stats import RATES_CSV, RateStats
from src.storage.db import RatesStore


class Decomposer:
    """STL decomposition and stationarity tests over a USD/MXN rate series
    (float values on a sorted DatetimeIndex with natural gaps)."""

    def __init__(self, series: pd.Series):
        self.series = series

    @classmethod
    def from_store(cls, store: RatesStore) -> "Decomposer":
        return cls(RateStats.from_store(store).series)

    def stl(self, period: int = 252, robust: bool = True) -> DecomposeResult:
        """STL decomposition into trend + seasonal + residual.

        `period` is in observations (trading days), since the index has gaps
        and no fixed frequency; 252 ~= one trading year.
        """
        return STL(self.series.to_numpy(), period=period, robust=robust).fit()

    def stl_strength(self, period: int = 252, robust: bool = True) -> dict:
        """Trend/seasonal strength (Hyndman's 0-1 measures) from an STL fit."""
        res = self.stl(period=period, robust=robust)
        resid_var = np.var(res.resid)
        trend_strength = max(0.0, 1 - resid_var / np.var(res.trend + res.resid))
        seasonal_strength = max(0.0, 1 - resid_var / np.var(res.seasonal + res.resid))
        return {
            "period": period,
            "trend_strength": float(trend_strength),
            "seasonal_strength": float(seasonal_strength),
        }

    def adf(self, series: pd.Series | None = None) -> dict:
        """Augmented Dickey-Fuller test. Null hypothesis: unit root
        (non-stationary) — low p-value means stationary."""
        values = (self.series if series is None else series).to_numpy()
        stat, pvalue, lags, _, critical, _ = adfuller(values, autolag="AIC")
        return {
            "test": "ADF",
            "statistic": float(stat),
            "pvalue": float(pvalue),
            "lags": int(lags),
            "critical_values": {k: float(v) for k, v in critical.items()},
            "stationary_at_5pct": pvalue < 0.05,
        }

    def kpss(self, series: pd.Series | None = None) -> dict:
        """KPSS test. Null hypothesis: stationary — low p-value means
        non-stationary (note the inverted null vs ADF). The reported p-value
        is clipped to statsmodels' table bounds [0.01, 0.1]."""
        values = (self.series if series is None else series).to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            stat, pvalue, lags, critical = kpss(values, regression="c", nlags="auto")
        return {
            "test": "KPSS",
            "statistic": float(stat),
            "pvalue": float(pvalue),
            "lags": int(lags),
            "critical_values": {k: float(v) for k, v in critical.items()},
            "stationary_at_5pct": pvalue > 0.05,
        }

    def stationarity_summary(self) -> dict:
        """ADF + KPSS on the rate levels and on their log returns.

        Expected outcome for FX: levels non-stationary (random-walk-like),
        returns stationary — which is why the forecast phase models returns.
        """
        returns = RateStats(self.series).log_returns()
        return {
            "levels": {"adf": self.adf(), "kpss": self.kpss()},
            "log_returns": {"adf": self.adf(returns), "kpss": self.kpss(returns)},
        }


def _print_test(result: dict) -> None:
    verdict = "stationary" if result["stationary_at_5pct"] else "non-stationary"
    print(
        f"  {result['test']}: statistic {result['statistic']:+.4f}, "
        f"p-value {result['pvalue']:.4f}, lags {result['lags']} -> {verdict} at 5%"
    )


def main() -> int:
    decomposer = Decomposer.from_store(RatesStore(RATES_CSV))

    summary = decomposer.stationarity_summary()
    print("Rate levels:")
    _print_test(summary["levels"]["adf"])
    _print_test(summary["levels"]["kpss"])
    print("Log returns:")
    _print_test(summary["log_returns"]["adf"])
    _print_test(summary["log_returns"]["kpss"])

    strength = decomposer.stl_strength()
    print(f"STL (period={strength['period']} trading days):")
    print(f"  trend strength: {strength['trend_strength']:.3f}   seasonal strength: {strength['seasonal_strength']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
