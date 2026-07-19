"""Daily collector: fetch the latest USD/MXN rate (Banxico primary, market
API fallback) and append it to data/rates.csv.

Email alerting is added in Phase 3 — this only fetches and stores.

Usage:
    python -m src.collector.run_daily
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.collector.fetch_banxico import BanxicoClient
from src.collector.fetch_fallback import FallbackClient
from src.storage.db import RatesStore

ROOT = Path(__file__).resolve().parent.parent.parent
RATES_CSV = ROOT / "data" / "rates.csv"


class DailyCollector:
    """Fetches the latest rate (Banxico, falling back to the market API) and
    appends it to a RatesStore."""

    def __init__(self, store: RatesStore, fallback: FallbackClient, banxico: BanxicoClient | None = None):
        self.store = store
        self.fallback = fallback
        self.banxico = banxico

    def fetch_latest(self) -> tuple[str, str, str]:
        """Return (date_iso, rate_str, source_name), trying Banxico then the fallback."""
        if self.banxico is not None:
            try:
                date_iso, rate = self.banxico.fetch_fix_rate()
                return date_iso, rate, "Banxico"
            except Exception as e:
                print(f"Banxico fetch failed ({e}), trying fallback...", file=sys.stderr)
        else:
            print("BANXICO_TOKEN not set, using fallback source...", file=sys.stderr)

        date_iso, rate = self.fallback.fetch_rate()
        return date_iso, rate, "fallback (Frankfurter)"

    def run(self) -> tuple[str, str, str, bool]:
        """Fetch the latest rate and store it. Returns (date_iso, rate, source, added)."""
        date_iso, rate, source = self.fetch_latest()
        added = self.store.append(date_iso, rate)
        return date_iso, rate, source, added


def main() -> int:
    load_dotenv(ROOT / ".env")

    token = os.environ.get("BANXICO_TOKEN")
    collector = DailyCollector(
        store=RatesStore(RATES_CSV),
        fallback=FallbackClient(),
        banxico=BanxicoClient(token) if token else None,
    )

    try:
        date_iso, rate, source, added = collector.run()
    except Exception as e:
        print(f"Error: all sources failed: {e}", file=sys.stderr)
        return 1

    rel_path = RATES_CSV.relative_to(ROOT)
    if added:
        print(f"[{source}] Added {date_iso},{rate} to {rel_path}")
    else:
        print(f"[{source}] {date_iso} already present in {rel_path} — no change")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
