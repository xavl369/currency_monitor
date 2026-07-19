"""Live view: current rate, day-over-day change, recent-window chart."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.data import DashboardData
from src.dashboard.palette import ChartTheme

WINDOWS = {"30 days": 30, "90 days": 90, "365 days": 365}
DEFAULT_WINDOW = "90 days"


class LivePage:
    """Current rate at a glance plus a selectable 30/90/365-day chart."""

    def __init__(self, data: DashboardData):
        self.data = data

    def render(self) -> None:
        st.title("USD/MXN — Live")

        series = self.data.series
        latest_date, latest = series.index[-1], float(series.iloc[-1])
        prev_date, prev = series.index[-2], float(series.iloc[-2])
        change = latest - prev
        pct = change / prev * 100

        choice = st.segmented_control(
            "Window", list(WINDOWS), default=DEFAULT_WINDOW, label_visibility="collapsed"
        )
        days = WINDOWS[choice or DEFAULT_WINDOW]
        window = series[series.index >= latest_date - pd.Timedelta(days=days)]
        window_change_pct = (latest / float(window.iloc[0]) - 1) * 100

        tiles = st.columns(4)
        tiles[0].metric(
            f"Rate — {latest_date:%Y-%m-%d}",
            f"{latest:.4f}",
            delta=f"{change:+.4f} ({pct:+.2f}%) vs {prev_date:%b %d}",
            delta_color="off",
        )
        tiles[1].metric(
            f"{days}-day change", f"{window_change_pct:+.2f}%", delta_color="off"
        )
        tiles[2].metric(
            f"{days}-day high",
            f"{float(window.max()):.4f}",
            delta=f"{window.idxmax():%Y-%m-%d}",
            delta_color="off",
        )
        tiles[3].metric(
            f"{days}-day low",
            f"{float(window.min()):.4f}",
            delta=f"{window.idxmin():%Y-%m-%d}",
            delta_color="off",
        )

        st.plotly_chart(self._chart(window), theme=None)
        st.caption(
            "MXN per USD — Banxico FIX rate (ECB reference on fallback days), "
            f"data through {latest_date:%Y-%m-%d}."
        )

    def _chart(self, window: pd.Series) -> go.Figure:
        theme = ChartTheme.current()
        fig = go.Figure(
            go.Scatter(
                x=window.index,
                y=window.to_numpy(),
                mode="lines",
                name="USD/MXN",
                line={"color": theme.series[0], "width": 2},
                hovertemplate="%{y:.4f}<extra></extra>",
            )
        )
        last_x, last_y = window.index[-1], float(window.iloc[-1])
        fig.add_trace(
            go.Scatter(
                x=[last_x],
                y=[last_y],
                mode="markers",
                marker={"color": theme.series[0], "size": 9},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_annotation(
            x=last_x,
            y=last_y,
            text=f"<b>{last_y:.4f}</b>",
            showarrow=False,
            yshift=14,
            xanchor="right",
            font={"color": theme.ink, "size": 13},
        )
        fig.update_layout(theme.layout(height=380, yaxis={"tickformat": ".2f"}))
        return fig
