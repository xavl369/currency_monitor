"""Banxico SIE API client — fetches the latest published USD/MXN FIX rate
(series SF43718, the official exchange rate).

Requires a free Banxico SIE API token: https://www.banxico.org.mx/SieAPIRest/service/v1/token
Put it in a .env file at the project root: BANXICO_TOKEN=your_token

Standalone usage (quick console check):
    python -m src.collector.fetch_banxico
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import requests


class BanxicoClient:
    """Client for Banxico's SIE API, scoped to the USD/MXN FIX rate series."""

    SERIES_ID = "SF43718"
    BASE_URL = "https://www.banxico.org.mx/SieAPIRest/service/v1/series"

    def __init__(self, token: str):
        self.token = token

    @property
    def url(self) -> str:
        return f"{self.BASE_URL}/{self.SERIES_ID}/datos/oportuno"

    def fetch_fix_rate(self) -> tuple[str, str]:
        """Return (date_iso, rate_str) for the most recently published FIX rate."""
        response = requests.get(self.url, headers={"Bmx-Token": self.token}, timeout=10)
        response.raise_for_status()
        payload = response.json()

        latest = payload["bmx"]["series"][0]["datos"][0]
        date_iso = datetime.strptime(latest["fecha"], "%d/%m/%Y").strftime("%Y-%m-%d")
        return date_iso, latest["dato"]


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

    token = os.environ.get("BANXICO_TOKEN")
    if not token:
        print(
            "Error: BANXICO_TOKEN is not set.\n"
            "Add it to .env at the project root (BANXICO_TOKEN=your_token) "
            'or set it directly: $env:BANXICO_TOKEN = "your_token"',
            file=sys.stderr,
        )
        return 1

    client = BanxicoClient(token)
    try:
        date_iso, rate = client.fetch_fix_rate()
    except requests.HTTPError as e:
        print(f"Error: Banxico API request failed: {e}", file=sys.stderr)
        return 1
    except (KeyError, IndexError):
        print("Error: unexpected response shape from Banxico API", file=sys.stderr)
        return 1

    print(f"USD/MXN FIX rate on {date_iso}: {rate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
