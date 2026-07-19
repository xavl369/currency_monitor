"""Convert the raw Currency.txt (tab-separated, DD-MM-YYYY dates) export into
data/historical_seed.csv (comma-separated, ISO 8601 dates).

Usage:
    python scripts/convert_currency_to_csv.py
"""

import csv
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "Currency.txt"
DEST = ROOT / "data" / "historical_seed.csv"


def convert(source: Path, dest: Path) -> int:
    with source.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        header_fields = [c.strip() for c in header if c.strip()]
        if header_fields != ["Date", "Value"]:
            raise ValueError(f"unexpected header: {header!r}")

        rows = []
        for line_no, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(f"line {line_no}: expected 2 fields, got {row!r}")
            date_str, rate_str = (field.strip() for field in row)
            try:
                date_iso = datetime.strptime(date_str, "%d-%m-%Y").strftime("%Y-%m-%d")
            except ValueError as e:
                raise ValueError(f"line {line_no}: bad date {date_str!r}") from e
            try:
                float(rate_str)
            except ValueError as e:
                raise ValueError(f"line {line_no}: bad rate {rate_str!r}") from e
            rows.append((date_iso, rate_str))

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "rate"])
        writer.writerows(rows)

    return len(rows)


if __name__ == "__main__":
    count = convert(SOURCE, DEST)
    print(f"Wrote {count} rows to {DEST.relative_to(ROOT)}")
