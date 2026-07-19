"""Daily email notifier: current USD/MXN rate, day-over-day change, and a
7-session trend chart, sent over SMTP.

Works with any SMTP service; defaults target Gmail (create an app password
at https://myaccount.google.com/apppasswords). Configuration comes from
environment variables (.env locally, GitHub Secrets in CI):

    SMTP_USER      required — SMTP login (your Gmail address)
    SMTP_PASSWORD  required — app password / SMTP key
    EMAIL_TO       required — recipient(s), comma-separated
    SMTP_HOST      optional, default smtp.gmail.com
    SMTP_PORT      optional, default 587 (STARTTLS)
    EMAIL_FROM     optional, default SMTP_USER

Usage (sends a real email from the current data/rates.csv, as a setup check):
    python -m src.email.notifier
"""

import os
import smtplib
import ssl
import sys
from email.message import EmailMessage
from email.utils import make_msgid
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
RATES_CSV = ROOT / "data" / "rates.csv"
DASHBOARD_URL = "https://currencymonitor-usd-mxn.streamlit.app/"

# Chart chrome (dataviz reference palette, light surface — email bodies are white).
SERIES = "#2a78d6"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


class TrendChart:
    """Renders the recent-rates line chart to a PNG for inline embedding."""

    def render(self, rows: list[tuple[str, str]]) -> bytes:
        """Render (date_iso, rate_str) rows to PNG bytes."""
        dates = [d[5:].replace("-", "/") for d, _ in rows]  # MM/DD, compact
        rates = [float(r) for _, r in rows]

        fig, ax = plt.subplots(figsize=(6.0, 2.8), dpi=160)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        x = range(len(rates))
        ax.plot(x, rates, color=SERIES, linewidth=2, marker="o", markersize=4, zorder=3)
        ax.plot(x[-1], rates[-1], color=SERIES, marker="o", markersize=7, zorder=4)
        ax.annotate(
            f"{rates[-1]:.4f}",
            (x[-1], rates[-1]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="right",
            fontsize=10,
            fontweight="bold",
            color=INK_PRIMARY,
        )

        ax.set_xticks(list(x), dates)
        ax.margins(x=0.04, y=0.18)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

        fig.tight_layout(pad=0.6)
        buf = BytesIO()
        fig.savefig(buf, format="png", facecolor="white")
        plt.close(fig)
        return buf.getvalue()


class EmailNotifier:
    """Composes and sends the daily rate email to a fixed recipient list."""

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        sender: str,
        recipients: list[str],
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.sender = sender
        self.recipients = recipients
        self.chart = TrendChart()

    @classmethod
    def from_env(cls) -> "EmailNotifier | None":
        """Build from environment variables; None if any required one is missing."""
        user = os.environ.get("SMTP_USER")
        password = os.environ.get("SMTP_PASSWORD")
        to = os.environ.get("EMAIL_TO")
        if not (user and password and to):
            return None
        return cls(
            host=os.environ.get("SMTP_HOST", "smtp.gmail.com"),
            port=int(os.environ.get("SMTP_PORT", "587")),
            user=user,
            password=password,
            sender=os.environ.get("EMAIL_FROM") or user,
            recipients=[a.strip() for a in to.split(",") if a.strip()],
        )

    def send_daily_update(self, rows: list[tuple[str, str]], source: str | None = None) -> None:
        """Send the daily email built from the full (date_iso, rate_str) history."""
        msg = self.compose(rows, source)
        with smtplib.SMTP(self.host, self.port, timeout=30) as smtp:
            smtp.starttls(context=ssl.create_default_context())
            smtp.login(self.user, self.password)
            smtp.send_message(msg)

    def compose(self, rows: list[tuple[str, str]], source: str | None = None) -> EmailMessage:
        """Build the full message without sending (also used by dry runs)."""
        if len(rows) < 2:
            raise ValueError("need at least two rows to report a day-over-day change")

        date_iso, rate = rows[-1][0], float(rows[-1][1])
        prev_date, prev = rows[-2][0], float(rows[-2][1])
        change = rate - prev
        pct = change / prev * 100
        arrow = "▲" if change > 0 else "▼" if change < 0 else "→"
        window = rows[-7:]

        msg = EmailMessage()
        msg["Subject"] = f"USD/MXN {date_iso}: {rate:.4f} ({arrow} {change:+.4f}, {pct:+.2f}%)"
        msg["From"] = self.sender
        msg["To"] = ", ".join(self.recipients)

        source_note = f"Source: {source}. " if source else ""
        table_text = "\n".join(f"  {d}  {float(r):.4f}" for d, r in window)
        msg.set_content(
            f"USD/MXN — {date_iso}\n\n"
            f"Rate: {rate:.4f} MXN per USD\n"
            f"Change vs {prev_date}: {change:+.4f} ({pct:+.2f}%)\n\n"
            f"Last {len(window)} sessions:\n{table_text}\n\n"
            f"Dashboard: {DASHBOARD_URL}\n\n"
            f"{source_note}Sent by usd-mxn-monitor."
        )

        cid = make_msgid()
        table_rows = "".join(
            f'<tr><td style="padding:2px 16px 2px 0; color:#52514e;">{d}</td>'
            f'<td style="padding:2px 0; text-align:right; font-variant-numeric:tabular-nums;">{float(r):.4f}</td></tr>'
            for d, r in window
        )
        msg.add_alternative(
            f"""\
<div style="font-family:system-ui,-apple-system,'Segoe UI',sans-serif; max-width:560px; margin:0 auto; padding:8px 0; color:{INK_PRIMARY};">
  <p style="font-size:13px; color:{INK_SECONDARY}; margin:0 0 2px;">USD/MXN &mdash; {date_iso}</p>
  <p style="font-size:34px; font-weight:600; margin:0;">{rate:.4f}</p>
  <p style="font-size:14px; color:{INK_SECONDARY}; margin:4px 0 16px;">{arrow} {change:+.4f} ({pct:+.2f}%) vs {prev_date}</p>
  <img src="cid:{cid[1:-1]}" width="520" alt="Line chart of the last {len(window)} sessions, closing at {rate:.4f}" style="max-width:100%; height:auto;"/>
  <table style="border-collapse:collapse; font-size:13px; margin:12px 0 16px;">{table_rows}</table>
  <p style="font-size:13px; margin:0 0 12px;"><a href="{DASHBOARD_URL}" style="color:{SERIES}; text-decoration:none;">View full dashboard &rarr;</a></p>
  <p style="font-size:12px; color:{INK_MUTED}; margin:0;">{source_note}Sent by usd-mxn-monitor.</p>
</div>
""",
            subtype="html",
        )
        msg.get_payload()[1].add_related(self.chart.render(window), "image", "png", cid=cid)
        return msg


def main() -> int:
    from dotenv import load_dotenv

    from src.storage.db import RatesStore

    load_dotenv(ROOT / ".env")

    notifier = EmailNotifier.from_env()
    if notifier is None:
        missing = [k for k in ("SMTP_USER", "SMTP_PASSWORD", "EMAIL_TO") if not os.environ.get(k)]
        print(f"Email not configured — missing: {', '.join(missing)} (see .env.example)", file=sys.stderr)
        return 1

    rows = RatesStore(RATES_CSV).read()
    notifier.send_daily_update(rows, source="manual test")
    print(f"Sent {rows[-1][0]} update to {', '.join(notifier.recipients)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
