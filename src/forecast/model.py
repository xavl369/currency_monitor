"""LSTM architecture for next-day USD/MXN scaled-log-return prediction, plus
the canonical paths of the saved model artifact and its metadata sidecar.

The Dropout layer doubles as the MC-dropout mechanism: `Predictor` calls the
model with training=True at inference to keep it active and sample the
predictive distribution. `recurrent_dropout` is deliberately not used — it is
slow on CPU and unnecessary for that purpose.
"""

from pathlib import Path

import keras

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "lstm_v1.keras"
META_PATH = MODELS_DIR / "lstm_v1.meta.json"


class ModelBuilder:
    """Builds the compiled Keras LSTM; all architecture hyperparameters
    default here, in one place."""

    def __init__(self, units: int = 64, dropout: float = 0.2, learning_rate: float = 1e-3):
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate

    def build(self, lookback: int) -> keras.Model:
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(lookback, 1)),
                keras.layers.LSTM(self.units),
                keras.layers.Dropout(self.dropout),
                keras.layers.Dense(1),
            ],
            name="lstm_v1",
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )
        return model
