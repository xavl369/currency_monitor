# USD/MXN Monitor

Personal, always-free tool that tracks the USD/MXN exchange rate end to end: a GitHub Actions cron job fetches the official daily rate (Banxico FIX, with an ECB/Frankfurter fallback), appends it to a git-committed CSV, and emails a daily update; a Streamlit dashboard serves live, historical, analytical, and LSTM-forecast views of the full 1991→present series.

## Architecture

```
fetch (Banxico → fallback) → RatesStore → data/rates.csv → email / analysis / forecast / dashboard
```

```
.github/workflows/   daily_fetch.yml (13:30 UTC weekdays), tests.yml (pytest on push/PR)
data/                historical_seed.csv (frozen), rates.csv (live store, bot-committed daily)
models/              lstm_v1.keras + lstm_v1.meta.json (trained artifact + frozen scaler/splits)
src/
├── collector/       BanxicoClient, FallbackClient, DailyCollector (the daily entrypoint)
├── storage/         RatesStore — CSV read + idempotent append
├── email/           EmailNotifier, TrendChart — daily SMTP update
├── analysis/        RateStats, Decomposer, BaselineForecaster (naive/ARIMA)
├── forecast/        ForecastPreprocessor, ModelBuilder, Trainer, Predictor (LSTM + MC dropout)
└── dashboard/       Streamlit app: Live / History / Analysis / Forecast pages
scripts/             one-off seed conversion + historical plot
tests/               pytest suite (storage, collector, fetch clients)
```

Invariants worth knowing before touching anything:

- **Dates are ISO 8601 (`YYYY-MM-DD`) everywhere in `data/*.csv`.** Source formats (`DD/MM/YYYY` from Banxico) are converted at the fetch boundary, never downstream.
- **`RatesStore.append` is the only writer of `data/rates.csv`** and is idempotent: an already-present date is a no-op (returns `False`), anything else is inserted and the file re-sorted. Re-running the collector is always safe.
- **`data/historical_seed.csv` never changes after creation.**
- Code is class-based by convention (`BanxicoClient`, `RatesStore`, …); new modules should follow suit.

## Setup

Python **3.11**, managed with [uv](https://docs.astral.sh/uv/):

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt      # full app (includes TensorFlow)
uv pip install --python .venv\Scripts\python.exe -r requirements-dev.txt  # test/CI extras (pytest)
```

There is no installed package — always run from the repo root as `python -m src...` so the `src` package resolves.

## Configuration / secrets

Copy `.env.example` to `.env` (gitignored) and fill in what you use:

| Variable | Required? | Purpose |
|---|---|---|
| `BANXICO_TOKEN` | Optional | Free token from [Banxico SIE](https://www.banxico.org.mx/SieAPIRest/service/v1/token). Without it the collector uses the Frankfurter/ECB fallback. |
| `SMTP_USER`, `SMTP_PASSWORD`, `EMAIL_TO` | Optional | Daily email. For Gmail, create an [app password](https://myaccount.google.com/apppasswords). If unset, the email is skipped with a log line — never an error. |
| `SMTP_HOST`, `SMTP_PORT`, `EMAIL_FROM` | Optional | Defaults: `smtp.gmail.com`, `587`, `EMAIL_FROM` = `SMTP_USER`. Any SMTP service works. |

The same names exist as **GitHub Actions secrets** for the daily workflow. Never commit `.env`.

## Commands

```powershell
# daily collector: fetch → append to data/rates.csv → email (if configured)
.\.venv\Scripts\python.exe -m src.collector.run_daily

# check one source only, no writes
.\.venv\Scripts\python.exe -m src.collector.fetch_banxico
.\.venv\Scripts\python.exe -m src.collector.fetch_fallback

# send a real test email using .env credentials
.\.venv\Scripts\python.exe -m src.email.notifier

# dashboard (from the repo root)
.\.venv\Scripts\python.exe -m streamlit run src/dashboard/app.py

# retrain the LSTM from data/rates.csv, or evaluate the saved artifact
.\.venv\Scripts\python.exe -m src.forecast.train
.\.venv\Scripts\python.exe -m src.forecast.predict

# tests
.\.venv\Scripts\python.exe -m pytest

# one-offs: regenerate seed CSV from Currency.txt / historical HTML chart
.\.venv\Scripts\python.exe scripts\convert_currency_to_csv.py
.\.venv\Scripts\python.exe scripts\plot_currency.py
```

## Automation

- **`daily_fetch.yml`** — 13:30 UTC weekdays (plus manual `workflow_dispatch`): runs the collector with the repo secrets and commits the updated `data/rates.csv` back. Banxico publishes with a lag, so a mid-day fetch often returns the previous business day — the idempotent append makes that a safe no-op, and the email is only sent when a row was actually added.
- **`tests.yml`** — runs the pytest suite on every push and PR, installing only `requirements-dev.txt` (no TensorFlow), so a run costs ~1 minute.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest
```

28 tests cover `RatesStore` integrity (idempotent append, dedupe, sorting, CRLF), `DailyCollector` orchestration (Banxico → fallback, `added` signalling) and both fetch clients' payload/date parsing with monkeypatched HTTP. No test touches the network or `data/` — all writes go to pytest temp dirs.

## Deploying the dashboard to Streamlit Community Cloud

The repo is deploy-ready; the one-time setup is interactive:

1. Push `main` to GitHub (private repos are supported).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and grant access to private repositories when prompted.
3. **Create app → Deploy from GitHub**: repository `xavl369/currency_monitor`, branch `main`, main file path `src/dashboard/app.py` (works as-is — the app adds the repo root to `sys.path` itself).
4. In **Advanced settings**, set the Python version to **3.11**.
5. Secrets: none needed — the dashboard only reads committed files (`data/rates.csv`, `models/lstm_v1.keras`).
6. Deploy. Every push to `main` — including the daily bot commit of `rates.csv` — redeploys automatically, so the hosted app stays current.
7. The app URL is public by default; restrict viewers in the app's settings if you want it private.

**Memory caveat:** Community Cloud gives roughly 1 GB of RAM and `requirements.txt` includes TensorFlow. The dashboard imports TF lazily — only when the Forecast page runs a prediction — so Live/History/Analysis pages are comfortable; opening Forecast may exceed the limit and restart the app. If that happens, swap `tensorflow` for `tensorflow-cpu` in `requirements.txt` (same API, smaller footprint), and if it still doesn't fit, treat the Forecast page as local-only.

## Data notes

- `.gitattributes` pins `data/*.csv` to **CRLF**: Python's `csv` writer emits CRLF on every platform (including the Linux runner), and pinning keeps the daily bot commit a clean one-row diff.
- Banxico's FIX rate publishes with a lag; querying mid-day usually returns the previous business day. That's normal, not an error.
- Full phase-by-phase history and design decisions: `USD_MXN_Monitor_ProjectSpec.md`.
