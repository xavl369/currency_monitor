"""Forecast view: pick a business date, see the LSTM estimate with its
MC-dropout 90% band, next to the naive and ARIMA baselines — including the
honest walk-forward comparison (for daily FX, expect the LSTM not to beat
the naive random walk)."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.data import FORECAST_SEED, DashboardData
from src.dashboard.palette import ChartTheme

MAX_HORIZON = 30  # business days
DEFAULT_HORIZON = 5
HISTORY_SESSIONS = 90  # context shown before the forecast fan


class ForecastPage:
    """Date-picked LSTM forecast with uncertainty band and baselines."""

    def __init__(self, data: DashboardData):
        self.data = data

    def render(self) -> None:
        st.title("Forecast")
        if not self.data.model_available():
            st.error(
                "No trained model found in models/ — run "
                "`python -m src.forecast.train` first."
            )
            return

        series = self.data.series
        last_date = series.index[-1]
        future = pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=MAX_HORIZON)

        picked = st.date_input(
            "Forecast date (business days only)",
            value=future[DEFAULT_HORIZON - 1].date(),
            min_value=future[0].date(),
            max_value=future[-1].date(),
        )
        target = pd.Timestamp(picked)
        if target.weekday() >= 5:  # date_input can't exclude weekends
            target = target - pd.offsets.BDay(1)
            st.caption(f"{picked:%Y-%m-%d} is a weekend — using {target:%Y-%m-%d}.")
        horizon = len(pd.bdate_range(start=future[0], end=target))

        forecast = self.data.forecast(horizon)
        baselines = self.data.baseline_forecasts(horizon)
        at_target = forecast.loc[target]

        tiles = st.columns(4)
        tiles[0].metric(
            f"LSTM — {target:%Y-%m-%d}",
            f"{at_target['mean']:.4f}",
            delta=f"{at_target['mean'] - float(series.iloc[-1]):+.4f} vs last close",
            delta_color="off",
        )
        tiles[1].metric(
            "90% band",
            f"{at_target['p05']:.3f}–{at_target['p95']:.3f}",  # compact — st.metric ellipsizes long values
        )
        tiles[2].metric("Naive (random walk)", f"{baselines.loc[target, 'naive']:.4f}")
        tiles[3].metric("ARIMA(1,1,1)", f"{baselines.loc[target, 'arima']:.4f}")

        st.plotly_chart(self._chart(series, forecast, baselines, target), theme=None)
        meta = self.data.model_meta()
        st.caption(
            f"LSTM trained {meta['created_utc'][:10]} on data through "
            f"{meta['series_end']}; band = MC dropout (200 samples, seed "
            f"{FORECAST_SEED}) + per-step residual noise. Business days only; "
            "MX/US holidays are not excluded."
        )

        self._evaluation_section()

    def _chart(
        self,
        series: pd.Series,
        forecast: pd.DataFrame,
        baselines: pd.DataFrame,
        target: pd.Timestamp,
    ) -> go.Figure:
        theme = ChartTheme.current()
        history = series.iloc[-HISTORY_SESSIONS:]
        last_date, last_rate = history.index[-1], float(history.iloc[-1])

        # prepend the last observation so forecast traces connect to history
        def joined(column: pd.Series) -> tuple[list, list]:
            return (
                [last_date, *column.index],
                [last_rate, *column.to_numpy()],
            )

        fig = go.Figure(
            go.Scatter(
                x=history.index,
                y=history.to_numpy(),
                mode="lines",
                name="Observed",
                line={"color": theme.series[0], "width": 2},
                hovertemplate="%{y:.4f}<extra>Observed</extra>",
            )
        )

        band_x, band_hi = joined(forecast["p95"])
        _, band_lo = joined(forecast["p05"])
        fig.add_trace(
            go.Scatter(
                x=band_x, y=band_hi, mode="lines", line={"width": 0},
                hoverinfo="skip", showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=band_x,
                y=band_lo,
                mode="lines",
                name="90% band",
                line={"width": 0},
                fill="tonexty",
                fillcolor=ChartTheme.rgba(theme.series[1], 0.15),
                hoverinfo="skip",
            )
        )

        mean_x, mean_y = joined(forecast["mean"])
        fig.add_trace(
            go.Scatter(
                x=mean_x,
                y=mean_y,
                mode="lines+markers",
                name="LSTM mean",
                line={"color": theme.series[1], "width": 2, "dash": "dash"},
                marker={"size": 5},
                hovertemplate="%{y:.4f}<extra>LSTM mean</extra>",
            )
        )
        naive_x, naive_y = joined(baselines["naive"])
        fig.add_trace(
            go.Scatter(
                x=naive_x,
                y=naive_y,
                mode="lines",
                name="Naive",
                line={"color": theme.muted, "width": 1.5, "dash": "dot"},
                hovertemplate="%{y:.4f}<extra>Naive</extra>",
            )
        )
        arima_x, arima_y = joined(baselines["arima"])
        fig.add_trace(
            go.Scatter(
                x=arima_x,
                y=arima_y,
                mode="lines",
                name="ARIMA(1,1,1)",
                line={"color": theme.series[2], "width": 1.5, "dash": "dot"},
                hovertemplate="%{y:.4f}<extra>ARIMA</extra>",
            )
        )

        fig.add_vline(x=last_date, line={"color": theme.baseline, "width": 1, "dash": "dash"})
        fig.add_vline(x=target, line={"color": theme.muted, "width": 1, "dash": "dot"})
        fig.update_layout(
            theme.layout(height=420, showlegend=True, yaxis={"tickformat": ".2f"})
        )
        return fig

    def _evaluation_section(self) -> None:
        st.subheader("Does the LSTM actually beat the baselines?")
        result = self.data.evaluation()

        labels = {
            "naive": "Naive random walk",
            "arima": f"ARIMA{tuple(result['order'])}",
            "lstm": "LSTM",
        }
        table = pd.DataFrame(
            {
                "Model": list(labels.values()),
                "MAE": [f"{result[k]['mae']:.4f}" for k in labels],
                "RMSE": [f"{result[k]['rmse']:.4f}" for k in labels],
                "MAPE": [f"{result[k]['mape_pct']:.3f}%" for k in labels],
            }
        )
        st.dataframe(table, hide_index=True)

        ratio = result["rmse_ratio_lstm_over_naive"]
        span = (
            f"walk-forward one-step-ahead over the last {result['n_test']} sessions "
            f"({result['test_start']} to {result['test_end']})"
        )
        if ratio < 1:
            st.success(
                f"The LSTM beats the naive random walk on RMSE "
                f"(ratio {ratio:.4f}), {span}."
            )
        else:
            st.warning(
                f"The LSTM does **not** beat the naive random walk on RMSE "
                f"(ratio {ratio:.4f}), {span}. For daily FX this is the "
                "expected result — treat the forecast as a scenario band "
                "around ‘no change’, not a predictive edge."
            )
