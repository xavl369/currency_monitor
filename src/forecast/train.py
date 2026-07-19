"""Training script for the USD/MXN LSTM: fits on the walk-forward training
span with early stopping on the validation span, saves the model artifact to
models/lstm_v1.keras plus a metadata sidecar (models/lstm_v1.meta.json) with
the frozen scaler, split configuration, and honest evaluation against the
Phase 4 baselines.

Standalone usage (train from data/rates.csv and save the artifact):
    python -m src.forecast.train
"""

import json
from datetime import datetime, timezone

import keras

from src.analysis.stats import RATES_CSV
from src.forecast.model import META_PATH, MODEL_PATH, MODELS_DIR, ModelBuilder
from src.forecast.predict import Predictor
from src.forecast.preprocess import ForecastPreprocessor
from src.storage.db import RatesStore


class Trainer:
    """Fits the LSTM on the training windows with early stopping on the
    validation span, then saves the artifact and metadata sidecar."""

    def __init__(
        self,
        preprocessor: ForecastPreprocessor,
        builder: ModelBuilder | None = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        seed: int = 42,
    ):
        self.preprocessor = preprocessor
        self.builder = builder or ModelBuilder()
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed

    def train(self) -> tuple[keras.Model, keras.callbacks.History]:
        keras.utils.set_random_seed(self.seed)
        model = self.builder.build(self.preprocessor.lookback)
        X_train, y_train = self.preprocessor.train_windows()
        X_val, y_val = self.preprocessor.val_windows()
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.patience,
                    restore_best_weights=True,
                )
            ],
            verbose=2,
        )
        return model, history

    def run(self) -> dict:
        """Train, save models/lstm_v1.keras, evaluate against the baselines
        (single implementation, in Predictor), and write the meta sidecar.
        Returns the metadata dict."""
        model, history = self.train()

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_PATH)

        predictor = Predictor(model, self.preprocessor)
        evaluation = predictor.evaluate()
        meta = {
            "model_file": MODEL_PATH.name,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": self.seed,
            "units": self.builder.units,
            "dropout": self.builder.dropout,
            "epochs_run": len(history.history["loss"]),
            "best_val_loss": float(min(history.history["val_loss"])),
            "residual_std_scaled": predictor.residual_std_scaled,
            **self.preprocessor.meta(),
            "evaluation": evaluation,
        }
        META_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return meta


def main() -> int:
    preprocessor = ForecastPreprocessor.from_store(RatesStore(RATES_CSV))
    meta = Trainer(preprocessor).run()

    print(f"Saved model:    {MODEL_PATH}")
    print(f"Saved metadata: {META_PATH}")
    print(f"Epochs run: {meta['epochs_run']} (best val loss {meta['best_val_loss']:.6f})")
    print(Predictor.format_evaluation(meta["evaluation"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
