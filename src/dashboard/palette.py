"""Chart palette and shared plotly chrome for the dashboard.

Colors are the dataviz reference palette (the same system the email chart in
`src/email/notifier.py` uses), with the dark column's re-stepped values —
not an automatic flip — selected when Streamlit renders in a dark theme.
Categorical slots are assigned in fixed order per chart and follow the
entity, never the series count: slot 1 is always the rate series.
"""

import streamlit as st

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

_LIGHT = {
    "series": ("#2a78d6", "#008300", "#e87ba4", "#eda100"),
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "baseline": "#c3c2b7",
}
_DARK = {
    "series": ("#3987e5", "#008300", "#d55181", "#c98500"),
    "ink": "#ffffff",
    "ink_secondary": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "baseline": "#383835",
}


class ChartTheme:
    """Palette slots plus a base plotly layout, resolved once per script run
    from the theme Streamlit is actually rendering in."""

    def __init__(self, dark: bool = False):
        colors = _DARK if dark else _LIGHT
        self.series: tuple[str, ...] = colors["series"]
        self.ink: str = colors["ink"]
        self.ink_secondary: str = colors["ink_secondary"]
        self.muted: str = colors["muted"]
        self.grid: str = colors["grid"]
        self.baseline: str = colors["baseline"]

    @classmethod
    def current(cls) -> "ChartTheme":
        theme = getattr(st.context, "theme", None)
        return cls(dark=theme is not None and theme.type == "dark")

    @staticmethod
    def rgba(hex_color: str, alpha: float) -> str:
        r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
        return f"rgba({r},{g},{b},{alpha})"

    def layout(self, **overrides) -> dict:
        """Base layout: transparent surfaces (the app background shows
        through), y-only hairline grid, recessive axes, unified hover."""
        layout = {
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {"family": FONT, "color": self.ink_secondary, "size": 13},
            "margin": {"l": 8, "r": 8, "t": 24, "b": 8},
            "hovermode": "x unified",
            "hoverlabel": {"font": {"family": FONT, "size": 12}},
            "showlegend": False,
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "x": 0,
                "font": {"color": self.ink_secondary},
            },
            "xaxis": {
                "showgrid": False,
                "linecolor": self.baseline,
                "tickcolor": self.baseline,
                "tickfont": {"color": self.muted, "size": 12},
                "automargin": True,
            },
            "yaxis": {
                "gridcolor": self.grid,
                "gridwidth": 1,
                "showline": False,
                "zeroline": False,
                "ticks": "",
                "tickfont": {"color": self.muted, "size": 12},
                "automargin": True,
            },
        }
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(layout.get(key), dict):
                layout[key] = {**layout[key], **value}
            else:
                layout[key] = value
        return layout
