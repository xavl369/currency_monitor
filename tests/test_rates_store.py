"""DB integrity tests for RatesStore (src/storage/db.py).

Every test writes only inside pytest's tmp_path; data/rates.csv is never
touched.
"""

from src.storage.db import RatesStore


def test_read_missing_file_returns_empty(store):
    assert store.read() == []


def test_get_last_date_none_when_empty(store):
    assert store.get_last_date() is None


def test_first_append_creates_file_with_header(store):
    assert store.append("2026-07-17", "18.6543") is True
    assert store.path.exists()
    assert store.path.read_text(encoding="utf-8").splitlines()[0] == "date,rate"
    assert store.read() == [("2026-07-17", "18.6543")]


def test_append_creates_parent_dirs(tmp_path):
    nested = RatesStore(tmp_path / "data" / "nested" / "rates.csv")
    assert nested.append("2026-07-17", "18.6543") is True
    assert nested.read() == [("2026-07-17", "18.6543")]


def test_duplicate_date_returns_false_and_keeps_original_rate(store):
    store.append("2026-07-17", "18.60")
    assert store.append("2026-07-17", "99.99") is False
    assert store.read() == [("2026-07-17", "18.60")]


def test_duplicate_append_leaves_file_byte_identical(store):
    store.append("2026-07-17", "18.60")
    before = store.path.read_bytes()
    assert store.append("2026-07-17", "18.61") is False
    assert store.path.read_bytes() == before


def test_out_of_order_append_resorts(store):
    store.append("2026-01-03", "17.30")
    store.append("2026-01-01", "17.10")
    store.append("2026-01-02", "17.20")
    assert store.read() == [
        ("2026-01-01", "17.10"),
        ("2026-01-02", "17.20"),
        ("2026-01-03", "17.30"),
    ]
    assert store.get_last_date() == "2026-01-03"


def test_header_written_exactly_once(store):
    store.append("2026-01-01", "17.10")
    store.append("2026-01-02", "17.20")
    store.append("2026-01-03", "17.30")
    text = store.path.read_text(encoding="utf-8")
    assert text.count("date,rate") == 1
    assert text.splitlines()[0] == "date,rate"


def test_rate_strings_round_trip_exactly(store):
    store.append("2026-01-01", "17.1234")
    store.append("2026-01-02", "18.50")
    assert store.read() == [("2026-01-01", "17.1234"), ("2026-01-02", "18.50")]


def test_file_uses_crlf_line_endings(store):
    # .gitattributes pins data/*.csv to CRLF; that only stays diff-free
    # because csv + newline="" emits CRLF on every platform, Linux included.
    store.append("2026-01-01", "17.10")
    store.append("2026-01-02", "17.20")
    raw = store.path.read_bytes()
    assert raw.count(b"\r\n") == 3  # header + 2 rows
    assert b"\n" not in raw.replace(b"\r\n", b"")  # no bare LF


def test_read_skips_blank_rows(store):
    store.path.write_text(
        "date,rate\r\n2026-01-01,17.10\r\n\r\n2026-01-02,17.20\r\n",
        encoding="utf-8",
        newline="",
    )
    assert store.read() == [("2026-01-01", "17.10"), ("2026-01-02", "17.20")]
