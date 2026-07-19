"""Forecasting from the trained LSTM artifact: one-step-ahead evaluation
against the Phase 4 baselines, and recursive multi-step forecasts with
MC-dropout uncertainty bands.

Evaluation is walk-forward one-step-ahead over the identical last-`test_size`
observations that `BaselineForecaster.evaluate` scores the naive/ARIMA
baselines on, with deterministic (dropout off) predictions — so the
naive/ARIMA/LSTM table is a like-for-like comparison. Be honest about the
result: for daily FX, an RMSE ratio near 1.0 against the naive random walk is
the expected outcome, and the dashboard should say so.

Standalone usage (evaluate the saved model and print a 5-day forecast):
    python -m src.forecast.predict
"""

import json
from pathlib import Path

import keras
import numpy as np
import pandas as pd

from src.analysis.baselines import BaselineForecaster
from src.analysis.stats import RATES_CSV
from src.forecast.model import META_PATH, MODEL_PATH
from src.forecast.preprocess import ForecastPreprocessor
from src.storage.db import RatesStore

MC_SAMPLES = 200
BAND_PERCENTILES = (5.0, 95.0)  # 90% band


class Predictor:
    """One-step and recursive multi-step forecasts from a trained LSTM, with
    MC-dropout bands, plus walk-forward evaluation against the baselines."""

    def __init__(
        self,
        model: keras.Model,
        preprocessor: ForecastPreprocessor,
        residual_std_scaled: float | None = None,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self._residual_std_scaled = residual_std_scaled

    @property
    def residual_std_scaled(self) -> float:
        """Std of the model's deterministic one-step errors (scaled-return
        space) on the validation span — the aleatoric noise per forecast step.
        Frozen into the meta sidecar at train time; computed lazily here only
        when constructed without one."""
        if self._residual_std_scaled is None:
            X_val, y_val = self.preprocessor.val_windows()
            pred = self.model.predict(X_val, verbose=0).ravel()
            self._residual_std_scaled = float(np.std(y_val - pred))
        return self._residual_std_scaled

    @classmethod
    def load(
        cls,
        model_path: Path = MODEL_PATH,
        meta_path: Path = META_PATH,
        store: RatesStore | None = None,
    ) -> "Predictor":
        """Load the saved artifact. The scaler comes frozen from the metadata
        (the model was trained on that scaling), never refit on current data."""
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"no trained model at {model_path} — run `python -m src.forecast.train` first"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        preprocessor = ForecastPreprocessor.from_store(
            store or RatesStore(RATES_CSV),
            lookback=meta["lookback"],
            val_size=meta["val_size"],
            test_size=meta["test_size"],
            scaler=(meta["scaler_mean"], meta["scaler_std"]),
        )
        return cls(
            keras.models.load_model(model_path),
            preprocessor,
            residual_std_scaled=meta.get("residual_std_scaled"),
        )

    def evaluate(self) -> dict:
        """Walk-forward one-step-ahead naive/ARIMA/LSTM comparison over the
        shared test span. Returns the baseline result dict extended with LSTM
        metrics and the LSTM/naive RMSE ratio (< 1 means the LSTM beat the
        random walk)."""
        X_test, _ = self.preprocessor.test_windows()
        levels = self.preprocessor.test_levels()

        pred_returns = self.preprocessor.unscale(
            self.model.predict(X_test, verbose=0).ravel()
        )
        lstm_pred = levels["prior_rate"].to_numpy() * np.exp(pred_returns)
        actual = levels["actual_rate"].to_numpy()
        lstm_metrics = BaselineForecaster._metrics(actual, lstm_pred)

        result = BaselineForecaster(self.preprocessor.series).evaluate(
            test_size=self.preprocessor.test_size
        )
        result["lstm"] = lstm_metrics
        result["rmse_ratio_lstm_over_naive"] = (
            lstm_metrics["rmse"] / result["naive"]["rmse"]
        )
        return result

    def forecast(
        self,
        horizon: int = 5,
        mc_samples: int = MC_SAMPLES,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Recursive multi-step forecast: each MC sample rolls the full
        horizon with dropout active (model uncertainty) plus per-step
        validation-residual noise (day-to-day aleatoric noise — MC dropout
        alone would give a misleadingly tight band around the model mean),
        feeding its own realized (scaled) return back into its window, so
        paths diffuse and bands widen ~sqrt(horizon). Returns rate levels
        indexed on the next `horizon` business days, columns mean/p05/p95."""
        rng = np.random.default_rng(seed)
        sigma = self.residual_std_scaled
        windows = np.repeat(self.preprocessor.latest_window(), mc_samples, axis=0)
        rates = np.full(mc_samples, self.preprocessor.last_rate)
        paths = np.empty((mc_samples, horizon))
        for step in range(horizon):
            pred_scaled = self.model(windows, training=True).numpy()  # (mc_samples, 1)
            sampled_scaled = (
                pred_scaled + rng.standard_normal((mc_samples, 1)) * sigma
            ).astype(np.float32)
            rates = rates * np.exp(self.preprocessor.unscale(sampled_scaled.ravel()))
            paths[:, step] = rates
            windows = np.concatenate(
                [windows[:, 1:, :], sampled_scaled[:, :, np.newaxis]], axis=1
            )

        lo, hi = BAND_PERCENTILES
        index = pd.bdate_range(
            start=self.preprocessor.last_date + pd.offsets.BDay(1),
            periods=horizon,
            name="date",
        )
        return pd.DataFrame(
            {
                "mean": paths.mean(axis=0),
                "p05": np.percentile(paths, lo, axis=0),
                "p95": np.percentile(paths, hi, axis=0),
            },
            index=index,
        )

    @staticmethod
    def format_evaluation(result: dict) -> str:
        """Console table for an `evaluate()` result (also stored in the meta
        sidecar, where tuples have become lists — hence the tuple() cast)."""
        lines = [
            f"Walk-forward one-step-ahead, last {result['n_test']} observations "
            f"({result['test_start']} to {result['test_end']}):"
        ]
        labels = {
            "naive": "Naive RW",
            "arima": f"ARIMA{tuple(result['order'])}",
            "lstm": "LSTM",
        }
        for key, label in labels.items():
            m = result[key]
            lines.append(
                f"  {label:>15}: MAE {m['mae']:.4f}   RMSE {m['rmse']:.4f}   MAPE {m['mape_pct']:.3f}%"
            )
        ratio = result["rmse_ratio_lstm_over_naive"]
        verdict = "beats" if ratio < 1 else "does not beat"
        lines.append(f"  LSTM {verdict} the naive random walk (RMSE ratio {ratio:.4f})")
        return "\n".join(lines)


def main() -> int:
    try:
        predictor = Predictor.load()
    except FileNotFoundError as exc:
        print(exc)
        return 1

    print(Predictor.format_evaluation(predictor.evaluate()))

    forecast = predictor.forecast()
    print(
        f"Next-{len(forecast)}-business-day LSTM forecast "
        f"(MC dropout, {MC_SAMPLES} samples, 90% band):"
    )
    for day, row in forecast.iterrows():
        print(
            f"  {day.strftime('%Y-%m-%d')}: {row['mean']:.4f}  "
            f"[{row['p05']:.4f}, {row['p95']:.4f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
