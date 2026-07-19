"""Streamlit entrypoint for the USD/MXN Monitor dashboard.

Run from the repo root:
    .venv\\Scripts\\python.exe -m streamlit run src/dashboard/app.py
"""

import sys
from datetime import date, timedelta
from pathlib import Path

# Streamlit puts this file's directory on sys.path, not the repo root — the
# root must be added before any `src` import resolves.
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st

from src.dashboard.data import DashboardData
from src.dashboard.views.analysis import AnalysisPage
from src.dashboard.views.forecast import ForecastPage
from src.dashboard.views.history import HistoryPage
from src.dashboard.views.live import LivePage

STALE_AFTER_BDAYS = 3


def sidebar(data: DashboardData) -> None:
    series = data.series
    last_date = series.index[-1]
    today = date.today()

    with st.sidebar:
        st.caption("Data freshness")
        st.markdown(f"**Data through {last_date:%Y-%m-%d}**")
        if last_date.date() < today:
            known = data.stats().get_last_known_value(today)
            if known is not None:
                st.markdown(f"As of today: **{known[1]:.4f}** (from {known[0]})")

        missed = int(
            np.busday_count(last_date.date() + timedelta(days=1), today + timedelta(days=1))
        )
        if missed > STALE_AFTER_BDAYS:
            st.warning(
                f"No new data for {missed} business days — check the "
                "daily-fetch workflow."
            )

        if data.model_available():
            meta = data.model_meta()
            st.caption(
                f"Forecast model trained {meta['created_utc'][:10]} "
                f"(data through {meta['series_end']})."
            )


def main() -> None:
    st.set_page_config(page_title="USD/MXN Monitor", page_icon="💱", layout="wide")

    data = DashboardData()
    nav = st.navigation(
        [
            st.Page(LivePage(data).render, title="Live", icon="📈", url_path="live", default=True),
            st.Page(HistoryPage(data).render, title="History", icon="🗓️", url_path="history"),
            st.Page(AnalysisPage(data).render, title="Analysis", icon="📊", url_path="analysis"),
            st.Page(ForecastPage(data).render, title="Forecast", icon="🔮", url_path="forecast"),
        ]
    )
    sidebar(data)
    nav.run()


if __name__ == "__main__":
    main()
