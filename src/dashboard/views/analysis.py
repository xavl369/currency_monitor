"""Analysis view: rolling volatility, STL decomposition, stationarity tests,
and an OLS regression trendline over a selectable window."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.dashboard.data import DashboardData
from src.dashboard.palette import ChartTheme

VOL_WINDOWS = {"21 sessions": 21, "63 sessions": 63, "252 sessions": 252}
DEFAULT_VOL = "21 sessions"

TREND_WINDOWS = {"1y": 252, "5y": 1260, "10y": 2520, "All": None}
DEFAULT_TREND = "5y"

STL_PERIOD = 252  # ~one trading year, in observations


class AnalysisPage:
    """Volatility, decomposition, stationarity, and trend regression."""

    def __init__(self, data: DashboardData):
        self.data = data

    def render(self) -> None:
        st.title("Analysis")
        self._volatility_section()
        self._regression_section()
        self._decomposition_section()
        self._stationarity_section()

    def _volatility_section(self) -> None:
        st.subheader("Rolling volatility")
        choice = st.segmented_control(
            "Volatility window",
            list(VOL_WINDOWS),
            default=DEFAULT_VOL,
            label_visibility="collapsed",
        )
        window = VOL_WINDOWS[choice or DEFAULT_VOL]
        vol = self.data.stats().rolling_volatility(window=window)

        theme = ChartTheme.current()
        fig = go.Figure(
            go.Scatter(
                x=vol.index,
                y=vol.to_numpy(),
                mode="lines",
                name="Annualized volatility",
                line={"color": theme.series[0], "width": 1.5},
                hovertemplate="%{y:.1%}<extra></extra>",
            )
        )
        fig.update_layout(theme.layout(height=340, yaxis={"tickformat": ".0%"}))
        st.plotly_chart(fig, theme=None)
        st.caption(
            f"Std of daily log returns over the last {window} sessions, "
            f"annualized by √252. Latest: {float(vol.iloc[-1]):.1%}."
        )

    def _regression_section(self) -> None:
        st.subheader("Regression trendline")
        choice = st.segmented_control(
            "Trend window",
            list(TREND_WINDOWS),
            default=DEFAULT_TREND,
            label_visibility="collapsed",
        )
        sessions = TREND_WINDOWS[choice or DEFAULT_TREND]
        series = self.data.series
        window = series if sessions is None else series.iloc[-sessions:]

        # OLS on days-since-start so the slope reads as MXN per year.
        x = (window.index - window.index[0]).days.to_numpy(dtype=float)
        y = window.to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1 - ss_res / ss_tot

        theme = ChartTheme.current()
        fig = go.Figure(
            go.Scatter(
                x=window.index,
                y=y,
                mode="lines",
                name="USD/MXN",
                line={"color": theme.series[0], "width": 1.5},
                hovertemplate="%{y:.4f}<extra>USD/MXN</extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=window.index,
                y=fitted,
                mode="lines",
                name="OLS trend",
                line={"color": theme.series[1], "width": 2, "dash": "dash"},
                hovertemplate="%{y:.4f}<extra>OLS trend</extra>",
            )
        )
        fig.update_layout(
            theme.layout(height=340, showlegend=True, yaxis={"tickformat": ".2f"})
        )
        st.plotly_chart(fig, theme=None)
        st.caption(
            f"OLS on calendar time over the selected window: slope "
            f"{slope * 365.25:+.3f} MXN/year, R² {r_squared:.3f}."
        )

    def _decomposition_section(self) -> None:
        st.subheader("STL decomposition")
        strength = self.data.stl_strength(STL_PERIOD)
        components = self.data.stl(STL_PERIOD)

        tiles = st.columns(3)
        tiles[0].metric("Period", f"{STL_PERIOD} sessions")
        tiles[1].metric("Trend strength", f"{strength['trend_strength']:.3f}")
        tiles[2].metric("Seasonal strength", f"{strength['seasonal_strength']:.3f}")

        theme = ChartTheme.current()
        titles = ("Observed", "Trend", "Seasonal", "Residual")
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            subplot_titles=titles,
        )
        for row, name in enumerate(titles, start=1):
            component = components[name.lower()]
            fig.add_trace(
                go.Scatter(
                    x=components.index,
                    y=component.to_numpy(),
                    mode="lines",
                    name=name,
                    line={"color": theme.series[0], "width": 1},
                    hovertemplate=f"%{{y:.4f}}<extra>{name}</extra>",
                ),
                row=row,
                col=1,
            )
        # shared axis styling has to go through update_x/yaxes to hit all 4 rows
        layout = theme.layout(height=720, hovermode="x")
        y_style = layout.pop("yaxis")
        x_style = layout.pop("xaxis")
        fig.update_layout(layout)
        fig.update_yaxes(**y_style)
        fig.update_xaxes(**x_style)
        fig.update_annotations(font={"color": theme.ink_secondary, "size": 13})
        st.plotly_chart(fig, theme=None)
        st.caption(
            "Hyndman 0-1 strength measures from a robust STL fit; for FX, "
            "expect strong trend and negligible seasonality."
        )

    def _stationarity_section(self) -> None:
        st.subheader("Stationarity")
        summary = self.data.stationarity()
        rows = []
        for series_label, key in (("Rate levels", "levels"), ("Log returns", "log_returns")):
            for test in ("adf", "kpss"):
                result = summary[key][test]
                rows.append(
                    {
                        "Series": series_label,
                        "Test": result["test"],
                        "Statistic": f"{result['statistic']:+.4f}",
                        "p-value": f"{result['pvalue']:.4f}",
                        "Lags": result["lags"],
                        "Verdict at 5%": "stationary"
                        if result["stationary_at_5pct"]
                        else "non-stationary",
                    }
                )
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(
            "ADF null: unit root (low p ⇒ stationary). KPSS null: stationary "
            "(low p ⇒ non-stationary; p clipped to [0.01, 0.1]). Expected for "
            "FX: levels non-stationary, returns stationary — which is why the "
            "forecast models returns, not levels."
        )
