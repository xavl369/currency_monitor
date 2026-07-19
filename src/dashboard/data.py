"""Cached data access for the Streamlit dashboard.

All Streamlit caching lives here, behind the `DashboardData` facade the view
pages are constructed with. Data caches are keyed on the mtimes of
data/rates.csv and the model meta sidecar, so the daily commit invalidates
them without an app restart. TensorFlow is imported lazily inside the
predictor loader so the other pages never pay its import cost.
"""

import json

import pandas as pd
import streamlit as st

from src.analysis.baselines import BaselineForecaster
from src.analysis.decomposition import Decomposer
from src.analysis.stats import RATES_CSV, RateStats
from src.forecast.model import META_PATH, MODEL_PATH
from src.storage.db import RatesStore

FORECAST_SEED = 42


@st.cache_data(show_spinner=False)
def _series(rates_mtime: float) -> pd.Series:
    return RateStats.from_store(RatesStore(RATES_CSV)).series


@st.cache_data(show_spinner="Running ADF/KPSS tests…")
def _stationarity(rates_mtime: float) -> dict:
    return Decomposer(_series(rates_mtime)).stationarity_summary()


@st.cache_data(show_spinner="Fitting STL decomposition…")
def _stl(rates_mtime: float, period: int) -> pd.DataFrame:
    series = _series(rates_mtime)
    res = Decomposer(series).stl(period=period)
    return pd.DataFrame(
        {
            "observed": series.to_numpy(),
            "trend": res.trend,
            "seasonal": res.seasonal,
            "residual": res.resid,
        },
        index=series.index,
    )


@st.cache_data(show_spinner="Computing trend/seasonal strength…")
def _stl_strength(rates_mtime: float, period: int) -> dict:
    return Decomposer(_series(rates_mtime)).stl_strength(period=period)


@st.cache_data(show_spinner="Fitting ARIMA baseline…")
def _baseline_forecasts(rates_mtime: float, horizon: int) -> pd.DataFrame:
    forecaster = BaselineForecaster(_series(rates_mtime))
    return pd.DataFrame(
        {
            "naive": forecaster.naive_forecast(horizon),
            "arima": forecaster.arima_forecast(horizon),
        }
    )


@st.cache_resource(show_spinner="Loading forecast model (TensorFlow)…")
def _predictor(rates_mtime: float, meta_mtime: float):
    from src.forecast.predict import Predictor  # lazy: pulls in TensorFlow

    return Predictor.load()


@st.cache_data(show_spinner="Evaluating LSTM vs baselines (walk-forward)…")
def _evaluation(rates_mtime: float, meta_mtime: float) -> dict:
    return _predictor(rates_mtime, meta_mtime).evaluate()


@st.cache_data(show_spinner="Sampling MC-dropout forecast paths…")
def _forecast(
    rates_mtime: float, meta_mtime: float, horizon: int, seed: int
) -> pd.DataFrame:
    return _predictor(rates_mtime, meta_mtime).forecast(horizon=horizon, seed=seed)


@st.cache_data(show_spinner=False)
def _model_meta(meta_mtime: float) -> dict:
    return json.loads(META_PATH.read_text(encoding="utf-8"))


class DashboardData:
    """Facade over the cached loaders; cheap to construct once per script run."""

    @property
    def _rates_mtime(self) -> float:
        return RATES_CSV.stat().st_mtime

    @property
    def _meta_mtime(self) -> float:
        return META_PATH.stat().st_mtime

    @property
    def series(self) -> pd.Series:
        return _series(self._rates_mtime)

    def stats(self) -> RateStats:
        return RateStats(self.series)

    def stationarity(self) -> dict:
        return _stationarity(self._rates_mtime)

    def stl(self, period: int = 252) -> pd.DataFrame:
        return _stl(self._rates_mtime, period)

    def stl_strength(self, period: int = 252) -> dict:
        return _stl_strength(self._rates_mtime, period)

    def baseline_forecasts(self, horizon: int) -> pd.DataFrame:
        return _baseline_forecasts(self._rates_mtime, horizon)

    def model_available(self) -> bool:
        return MODEL_PATH.exists() and META_PATH.exists()

    def model_meta(self) -> dict:
        return _model_meta(self._meta_mtime)

    def evaluation(self) -> dict:
        return _evaluation(self._rates_mtime, self._meta_mtime)

    def forecast(self, horizon: int, seed: int = FORECAST_SEED) -> pd.DataFrame:
        return _forecast(self._rates_mtime, self._meta_mtime, horizon, seed)
