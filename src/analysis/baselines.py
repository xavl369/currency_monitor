"""Forecast baselines for the USD/MXN series: naive random walk and ARIMA.

These exist to be honest yardsticks for the Phase 5 RNN — if the RNN can't
beat the naive random walk out-of-sample, the dashboard should say so.

Evaluation is walk-forward one-step-ahead over the last `test_size`
observations, with no lookahead: the naive forecast for day t is the observed
rate at t-1, and ARIMA parameters are estimated on the training span only,
then filtered forward (no refit) so each test-day prediction uses only data
through the previous observation.

Standalone usage (evaluate both baselines against data/rates.csv):
    python -m src.analysis.baselines
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.analysis.stats import RATES_CSV, RateStats
from src.storage.db import RatesStore


class BaselineForecaster:
    """Naive random-walk and ARIMA baselines over a USD/MXN rate series
    (float values on a sorted DatetimeIndex with natural gaps)."""

    def __init__(self, series: pd.Series, order: tuple[int, int, int] = (1, 1, 1)):
        self.series = series
        self.order = order

    @classmethod
    def from_store(cls, store: RatesStore, order: tuple[int, int, int] = (1, 1, 1)) -> "BaselineForecaster":
        return cls(RateStats.from_store(store).series, order=order)

    def _future_index(self, horizon: int) -> pd.DatetimeIndex:
        """Business days after the last observation (MX/US holidays not excluded)."""
        start = self.series.index[-1] + pd.offsets.BDay(1)
        return pd.bdate_range(start=start, periods=horizon, name="date")

    def naive_forecast(self, horizon: int = 1) -> pd.Series:
        """Random walk: every future day forecasts the last observed rate."""
        return pd.Series(
            np.full(horizon, self.series.iloc[-1]),
            index=self._future_index(horizon),
            name="naive",
        )

    def arima_forecast(self, horizon: int = 1) -> pd.Series:
        """ARIMA fit on the full series, forecast `horizon` steps ahead."""
        fitted = ARIMA(self.series.to_numpy(), order=self.order).fit()
        return pd.Series(
            fitted.forecast(steps=horizon),
            index=self._future_index(horizon),
            name=f"arima{self.order}",
        )

    def evaluate(self, test_size: int = 250) -> dict:
        """Walk-forward one-step-ahead comparison over the last `test_size`
        observations. Returns MAE/RMSE/MAPE per model plus the ARIMA/naive
        RMSE ratio (< 1 means ARIMA beat the random walk)."""
        n = self.series.size
        if not 0 < test_size <= n - 30:
            raise ValueError(f"test_size must leave >= 30 training observations (got {test_size} of {n})")

        values = self.series.to_numpy()
        actual = values[n - test_size:]

        naive_pred = values[n - test_size - 1: n - 1]

        trained = ARIMA(values[: n - test_size], order=self.order).fit()
        filtered = trained.apply(values)  # train-only params over the full series, no refit
        arima_pred = filtered.predict(start=n - test_size, end=n - 1)

        naive_metrics = self._metrics(actual, naive_pred)
        arima_metrics = self._metrics(actual, arima_pred)
        return {
            "test_start": self.series.index[n - test_size].strftime("%Y-%m-%d"),
            "test_end": self.series.index[-1].strftime("%Y-%m-%d"),
            "n_test": test_size,
            "order": self.order,
            "naive": naive_metrics,
            "arima": arima_metrics,
            "rmse_ratio_arima_over_naive": arima_metrics["rmse"] / naive_metrics["rmse"],
        }

    @staticmethod
    def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
        error = predicted - actual
        return {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mape_pct": float(np.mean(np.abs(error / actual)) * 100),
        }


def main() -> int:
    forecaster = BaselineForecaster.from_store(RatesStore(RATES_CSV))

    result = forecaster.evaluate()
    print(
        f"Walk-forward one-step-ahead, last {result['n_test']} observations "
        f"({result['test_start']} to {result['test_end']}):"
    )
    for model in ("naive", "arima"):
        m = result[model]
        label = f"ARIMA{result['order']}" if model == "arima" else "Naive RW"
        print(f"  {label:>15}: MAE {m['mae']:.4f}   RMSE {m['rmse']:.4f}   MAPE {m['mape_pct']:.3f}%")
    ratio = result["rmse_ratio_arima_over_naive"]
    verdict = "beats" if ratio < 1 else "does not beat"
    print(f"  ARIMA {verdict} the naive random walk (RMSE ratio {ratio:.4f})")

    print("Next-5-business-day forecasts (from full series):")
    naive = forecaster.naive_forecast(horizon=5)
    arima = forecaster.arima_forecast(horizon=5)
    for day in naive.index:
        print(f"  {day.strftime('%Y-%m-%d')}: naive {naive[day]:.4f}   arima {arima[day]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
