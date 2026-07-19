"""CSV-backed storage for USD/MXN rates: read, and idempotent, deduped,
sorted append.
"""

import csv
from pathlib import Path


class RatesStore:
    """CSV-backed store of (date, rate) rows, scoped to a single CSV path."""

    HEADER = ["date", "rate"]

    def __init__(self, path: Path):
        self.path = path

    def read(self) -> list[tuple[str, str]]:
        """Return all (date_iso, rate_str) rows, or [] if the file doesn't exist yet."""
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            return [(row[0], row[1]) for row in reader if row]

    def get_last_date(self) -> str | None:
        rows = self.read()
        return rows[-1][0] if rows else None

    def append(self, date_iso: str, rate: str) -> bool:
        """Insert (date_iso, rate), keeping the file deduped and sorted by date.

        Returns True if a row was written, False if date_iso was already present (no-op).
        """
        rows = self.read()
        if any(d == date_iso for d, _ in rows):
            return False

        rows.append((date_iso, rate))
        rows.sort(key=lambda r: r[0])

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADER)
            writer.writerows(rows)
        return True
