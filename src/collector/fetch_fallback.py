"""Backup USD/MXN rate source, used when Banxico is unavailable.

Frankfurter (https://api.frankfurter.app) mirrors ECB reference rates,
is free, and needs no API key/token.

Standalone usage (quick console check):
    python -m src.collector.fetch_fallback
"""

import sys

import requests


class FallbackClient:
    """Client for the Frankfurter API, scoped to the USD->MXN reference rate."""

    URL = "https://api.frankfurter.app/latest"

    def fetch_rate(self) -> tuple[str, str]:
        """Return (date_iso, rate_str) for the latest USD/MXN reference rate."""
        response = requests.get(self.URL, params={"from": "USD", "to": "MXN"}, timeout=10)
        response.raise_for_status()
        payload = response.json()

        date_iso = payload["date"]  # already YYYY-MM-DD
        rate = payload["rates"]["MXN"]
        return date_iso, str(rate)


def main() -> int:
    client = FallbackClient()
    try:
        date_iso, rate = client.fetch_rate()
    except requests.HTTPError as e:
        print(f"Error: fallback API request failed: {e}", file=sys.stderr)
        return 1
    except KeyError:
        print("Error: unexpected response shape from fallback API", file=sys.stderr)
        return 1

    print(f"USD/MXN rate on {date_iso}: {rate} (source: Frankfurter/ECB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
