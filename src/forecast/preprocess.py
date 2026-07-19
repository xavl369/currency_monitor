"""Windowing, scaling, and walk-forward splits for the USD/MXN LSTM forecast.

The model's target is the next-day log return, never the rate level; rate
forecasts are reconstructed downstream as rate_t = rate_{t-1} * exp(r_t).

Splits are chronological (walk-forward): the last `test_size` observations are
the test span — deliberately the same span `BaselineForecaster.evaluate`
scores the naive/ARIMA baselines on, so metrics are directly comparable — the
`val_size` observations before it drive early stopping, and everything earlier
is training data. Scaling is plain standardization whose mean/std come from
the training span only, so nothing from the validation or test spans leaks
into model inputs; at predict time the training-time scaler is passed back in
frozen via `scaler=` rather than refit.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.analysis.stats import RateStats
from src.storage.db import RatesStore


class ForecastPreprocessor:
    """Lagged log-return windows over a USD/MXN rate series (float values on
    a sorted DatetimeIndex with natural gaps), with train-only scaling and
    chronological train/val/test splits."""

    MIN_TRAIN_SAMPLES = 500

    def __init__(
        self,
        series: pd.Series,
        lookback: int = 60,
        val_size: int = 250,
        test_size: int = 250,
        scaler: tuple[float, float] | None = None,
    ):
        self.series = series
        self.lookback = lookback
        self.val_size = val_size
        self.test_size = test_size

        self.returns = RateStats(series).log_returns()
        n = self.returns.size
        self._val_start = n - test_size - val_size
        self._test_start = n - test_size
        if self._val_start - lookback < self.MIN_TRAIN_SAMPLES:
            raise ValueError(
                f"series too short: {n} returns leave fewer than "
                f"{self.MIN_TRAIN_SAMPLES} training samples after "
                f"lookback={lookback}, val_size={val_size}, test_size={test_size}"
            )

        if scaler is None:
            train_span = self.returns.to_numpy()[: self._val_start]
            scaler = (float(train_span.mean()), float(train_span.std()))
        self.scaler_mean, self.scaler_std = scaler
        self._scaled = (
            (self.returns.to_numpy() - self.scaler_mean) / self.scaler_std
        ).astype(np.float32)

    @classmethod
    def from_store(cls, store: RatesStore, **kwargs) -> "ForecastPreprocessor":
        return cls(RateStats.from_store(store).series, **kwargs)

    def unscale(self, values: np.ndarray) -> np.ndarray:
        """Map scaled model outputs back to log returns."""
        return np.asarray(values, dtype=np.float64) * self.scaler_std + self.scaler_mean

    def _windows(self, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        """(X, y) for target return indices [start, end): X is the `lookback`
        scaled returns before each target, shaped (n, lookback, 1)."""
        sw = sliding_window_view(self._scaled, self.lookback)
        X = sw[start - self.lookback : end - self.lookback][..., np.newaxis]
        return X, self._scaled[start:end]

    def train_windows(self) -> tuple[np.ndarray, np.ndarray]:
        return self._windows(self.lookback, self._val_start)

    def val_windows(self) -> tuple[np.ndarray, np.ndarray]:
        return self._windows(self._val_start, self._test_start)

    def test_windows(self) -> tuple[np.ndarray, np.ndarray]:
        return self._windows(self._test_start, self.returns.size)

    def test_levels(self) -> pd.DataFrame:
        """Per test day: the prior observed rate (for converting a predicted
        return to a level) and the actual rate. Indexed by the test dates —
        the same last-`test_size` observations the baselines are scored on."""
        values = self.series.to_numpy()
        n = self.series.size
        return pd.DataFrame(
            {
                "prior_rate": values[n - self.test_size - 1 : n - 1],
                "actual_rate": values[n - self.test_size :],
            },
            index=self.series.index[n - self.test_size :],
        )

    def latest_window(self) -> np.ndarray:
        """The most recent `lookback` scaled returns, shaped (1, lookback, 1)
        for one-step inference from the end of the series."""
        return self._scaled[-self.lookback :].reshape(1, self.lookback, 1)

    @property
    def last_rate(self) -> float:
        return float(self.series.iloc[-1])

    @property
    def last_date(self) -> pd.Timestamp:
        return self.series.index[-1]

    def meta(self) -> dict:
        """Everything predict-time loading needs to rebuild this preprocessor
        with the training-time scaling frozen."""
        return {
            "lookback": self.lookback,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "n_observations": int(self.series.size),
            "series_start": self.series.index[0].strftime("%Y-%m-%d"),
            "series_end": self.series.index[-1].strftime("%Y-%m-%d"),
        }
