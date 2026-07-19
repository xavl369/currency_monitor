"""Historical explorer: full 1991-present series, zoomable, with moving
averages and annotated crisis periods."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis.stats import RateStats
from src.dashboard.data import DashboardData
from src.dashboard.palette import ChartTheme

# (label, start, end) — approximate spans, for orientation not precision.
CRISES = [
    ("Tequila crisis", "1994-12-19", "1995-12-29"),
    ("Global financial crisis", "2008-09-01", "2009-03-31"),
    ("US election", "2016-11-08", "2017-01-20"),
    ("COVID-19 shock", "2020-02-20", "2020-04-30"),
    ("MX election / US tariffs", "2024-06-03", "2025-04-30"),
]

# Fixed categorical slots: the rate is always slot 1, each MA keeps its own
# slot regardless of which subset is toggled on (color follows the entity).
MOVING_AVERAGES = {"MA 30": (30, 1), "MA 90": (90, 2), "MA 200": (200, 3)}


class HistoryPage:
    """Full-history chart with rangeslider, MA toggles, and crisis shading."""

    def __init__(self, data: DashboardData):
        self.data = data

    def render(self) -> None:
        st.title("Historical explorer")

        controls = st.columns([3, 1])
        selected_mas = controls[0].pills(
            "Moving averages (sessions)",
            list(MOVING_AVERAGES),
            selection_mode="multi",
        )
        show_crises = controls[1].toggle("Crisis periods", value=True)

        series = self.data.series
        stats = self.data.stats()
        st.plotly_chart(
            self._chart(series, stats, selected_mas or [], show_crises), theme=None
        )
        st.caption(
            f"{series.size} sessions, {series.index[0]:%Y-%m-%d} to "
            f"{series.index[-1]:%Y-%m-%d}. Weekend/holiday gaps are real — "
            "no trading, no fabricated points."
        )

        with st.expander("Data table"):
            table = series.to_frame("rate")
            table.index = table.index.strftime("%Y-%m-%d")
            for label, (sessions, _) in MOVING_AVERAGES.items():
                if label in (selected_mas or []):
                    # positional (numpy) assignment — Series assignment
                    # index-aligns, which raises on the seed's duplicate dates
                    table[label] = table["rate"].rolling(sessions).mean().to_numpy()
            st.dataframe(
                table.sort_index(ascending=False).style.format("{:.4f}", na_rep="—"),
                height=360,
            )

    def _chart(
        self,
        series: pd.Series,
        stats: RateStats,
        selected_mas: list[str],
        show_crises: bool,
    ) -> go.Figure:
        theme = ChartTheme.current()
        fig = go.Figure(
            go.Scatter(
                x=series.index,
                y=series.to_numpy(),
                mode="lines",
                name="USD/MXN",
                line={"color": theme.series[0], "width": 1.5},
                hovertemplate="%{y:.4f}<extra>USD/MXN</extra>",
            )
        )
        for label, (sessions, slot) in MOVING_AVERAGES.items():
            if label not in selected_mas:
                continue
            ma = stats.rolling_mean(window=sessions)
            fig.add_trace(
                go.Scatter(
                    x=ma.index,
                    y=ma.to_numpy(),
                    mode="lines",
                    name=label,
                    line={"color": theme.series[slot], "width": 2},
                    hovertemplate=f"%{{y:.4f}}<extra>{label}</extra>",
                )
            )

        if show_crises:
            for label, start, end in CRISES:
                fig.add_vrect(
                    x0=start,
                    x1=end,
                    fillcolor=ChartTheme.rgba(theme.baseline, 0.22),
                    line_width=0,
                    annotation_text=label,
                    annotation_position="top left",
                    annotation_textangle=270,
                    annotation_font={"color": theme.muted, "size": 11},
                )

        fig.update_layout(
            theme.layout(
                height=520,
                showlegend=bool(selected_mas),
                yaxis={"tickformat": ".2f"},
                xaxis={
                    "rangeslider": {
                        "visible": True,
                        "thickness": 0.06,
                        "bordercolor": theme.baseline,
                        "borderwidth": 1,
                    },
                    "rangeselector": {
                        # top-right, clear of the top-left legend
                        "x": 1.0,
                        "xanchor": "right",
                        "y": 1.02,
                        "yanchor": "bottom",
                        "bgcolor": "rgba(0,0,0,0)",
                        "activecolor": ChartTheme.rgba(theme.baseline, 0.4),
                        "font": {"color": theme.ink_secondary, "size": 12},
                        "buttons": [
                            {"count": 1, "label": "1y", "step": "year"},
                            {"count": 5, "label": "5y", "step": "year"},
                            {"count": 10, "label": "10y", "step": "year"},
                            {"step": "all", "label": "All"},
                        ],
                    },
                },
            )
        )
        return fig
