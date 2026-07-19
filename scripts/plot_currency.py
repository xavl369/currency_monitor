"""Plot the USD/MXN historical series from data/historical_seed.csv.

Produces an interactive Plotly chart (zoom/pan, range slider, range-selector
buttons for 1Y/5Y/10Y/All) and writes it to plots/usd_mxn_history.html.

Usage:
    python scripts/plot_currency.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "historical_seed.csv"
DEST = ROOT / "plots" / "usd_mxn_history.html"


def plot(source: Path, dest: Path) -> None:
    df = pd.read_csv(source, parse_dates=["date"])

    fig = go.Figure(
        go.Scatter(
            x=df["date"],
            y=df["rate"],
            mode="lines",
            name="USD/MXN",
            line=dict(width=1.5),
        )
    )
    fig.update_layout(
        title="USD/MXN Exchange Rate (1991-2026)",
        xaxis_title="Date",
        yaxis_title="MXN per USD",
        template="plotly_white",
        hovermode="x unified",
    )
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(count=10, label="10Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ]
        ),
    )

    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(dest)


if __name__ == "__main__":
    plot(SOURCE, DEST)
    print(f"Wrote {DEST.relative_to(ROOT)}")
