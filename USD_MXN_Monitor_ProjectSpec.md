# USD/MXN Monitor — Project Spec

**Purpose:** Free, always-on tool to collect, store, analyze, and forecast the USD/MXN exchange rate, with a daily email alert and a Streamlit dashboard.

**Owner/architect:** Claude (guide) | **Builder:** Claude Code

---

## 1. High-level architecture

```
┌─────────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│ GitHub Actions cron  │─────▶│  Data store       │◀────▶│  Streamlit app   │
│ (daily fetch job)    │      │  (SQLite/Postgres)│      │  (dashboard)     │
└──────────┬───────────┘      └──────────────────┘      └─────────────────┘
           │
           ▼
   ┌───────────────┐
   │ Email alert   │
   │ (SMTP)        │
   └───────────────┘
```

- **Collector**: Python script, run daily by GitHub Actions, fetches the rate (Banxico SIE API primary, fallback market API), appends to the data store, sends the email.
- **Store**: start with a CSV file committed to the repo (`data/rates.csv`) — simplest, free, versioned, human-readable, and diffable in git history. At this scale (~8,700 rows over 34 years, one append/day) a CSV is perfectly sufficient. Leave a clean storage interface (`storage/db.py`) so it can swap to SQLite/Postgres later without touching the rest of the code.
- **Dashboard**: Streamlit app, reads from the same store, run locally now, deployable free later to Streamlit Community Cloud.
- **Forecast module**: RNN (Keras/TF, LSTM) trained offline, loaded into the dashboard for on-demand "pick a date" estimates with confidence bands, benchmarked against naive/ARIMA baselines.

---

## 2. Repo structure

```
usd-mxn-monitor/
├── .github/
│   └── workflows/
│       └── daily_fetch.yml        # cron trigger, calls collector
├── data/
│   ├── historical_seed.csv        # your existing 1991-2026 file, one-time import
│   └── rates.csv                  # CSV, updated daily, committed by Action
├── src/
│   ├── collector/
│   │   ├── fetch_banxico.py       # Banxico SIE API client
│   │   ├── fetch_fallback.py      # backup market API client
│   │   └── run_daily.py           # orchestrates fetch -> store -> email
│   ├── storage/
│   │   └── db.py                  # CSV interface (get/insert/query), idempotent append
│   ├── email/
│   │   └── notifier.py            # SMTP send, templated message
│   ├── analysis/
│   │   ├── stats.py               # returns, volatility, rolling stats
│   │   ├── decomposition.py       # STL, stationarity tests
│   │   └── baselines.py           # naive + ARIMA/GARCH for comparison
│   ├── forecast/
│   │   ├── preprocess.py          # windowing, scaling, train/val split (walk-forward)
│   │   ├── model.py               # LSTM/GRU architecture
│   │   ├── train.py               # training script, saves model artifact
│   │   └── predict.py             # loads model, recursive multi-step forecast + MC dropout bands
│   └── dashboard/
│       └── app.py                 # Streamlit entrypoint
├── models/
│   └── lstm_v1.keras              # trained model artifact
├── tests/
├── requirements.txt
├── .env.example                   # SMTP creds, API keys (never commit real .env)
└── README.md
```

---

## 3. Build order (phases for Claude Code)

**Phase 1 — Data foundation**
- Copy `historical_seed.csv` (your existing 1991-2026 file) to `data/rates.csv` as the starting point.
- Build `fetch_banxico.py` (Banxico SIE API, series SF43718 = FIX rate) with `fetch_fallback.py` as backup.
- Build `db.py` with idempotent append to the CSV (check last date / dedupe before writing, keep sorted by date).
- **Weekend/holiday guard**: `fetch_banxico.py` must check that the date returned by the API matches *today's* date before treating it as new data. Banxico doesn't publish on weekends or market holidays (Mexican or US), and some APIs silently return the last published (stale) value instead of erroring — without this check, that stale value could get written as if it were a new row, corrupting stats and RNN training data. If the returned date isn't today, skip the insert and skip sending the email that day.
- Manual test: run collector once locally, confirm new row appears at the end of `rates.csv`.

**Phase 2 — Automation**
- Write `daily_fetch.yml` GitHub Actions workflow (cron schedule, e.g. `30 13 * * 1-5` UTC to catch Banxico's daily publish, **weekdays only** — no point running Sat/Sun since FX doesn't trade and no new rate is published; combined with the weekend/holiday guard in `fetch_banxico.py`, this also protects against weekday market holidays).
- Store SMTP credentials as GitHub Secrets.
- Workflow: checkout → run `run_daily.py` → commit updated `rates.csv` back to repo.
- Test with manual workflow_dispatch trigger before trusting the schedule.

**Phase 3 — Email alerts**
- `notifier.py`: simple daily email — current rate, day-over-day change, 7-day trend line as inline chart or attached image.
- Use Gmail app password or SendGrid free tier (500 emails/day free — way more than needed).

**Phase 4 — Analysis module**
- Descriptive stats, rolling volatility, STL decomposition, stationarity tests (ADF/KPSS) — reusable functions, not UI yet.
- Naive random-walk and ARIMA baselines for later comparison against the RNN.
- Note: don't forward-fill weekend/holiday gaps in the dataset — pandas/statsmodels handle a datetime index with gaps naturally, and filling them would falsely imply "no change" on days that never traded.
- **Display helper**: add a `get_last_known_value(date)` function that, given any date (including weekends/holidays), looks backward and returns the most recent available rate — used only by the dashboard and email display layer (Phase 3, Phase 6), never written back into `rates.csv`. This keeps "what's the dollar worth today" sensible on a Saturday without fabricating a data point in the actual dataset.

**Phase 5 — Forecast (RNN)**
- Preprocessing: lagged windows, return-based target, walk-forward split.
- LSTM/GRU model with MC-dropout for uncertainty bands.
- Training script + saved artifact.
- Evaluate against baselines from Phase 4 — be honest in the dashboard about how much (or little) it beats naive.

**Phase 6 — Dashboard (Streamlit)**
- Page 1: Live view — current rate, day change, last 30/90/365 days chart (plotly).
- Page 2: Historical explorer — full 1991-2026 series, zoomable, crisis periods annotated, moving averages toggle.
- Page 3: Analysis — volatility charts, decomposition, regression trendlines.
- Page 4: Forecast — date picker, shows RNN estimate + confidence band + baseline comparison.
- Sidebar: data freshness indicator (last updated date).

**Phase 7 — Polish**
- README with setup instructions.
- `.env.example` and secrets documentation.
- Basic tests for collector idempotency and DB integrity.
- Optional: deploy dashboard to Streamlit Community Cloud (free).

**Phase 8 — Future: multi-user web product (not now — reference only)**

*Do not build this alongside Phases 1-7. This is a separate future project that reuses the core engine (collector, analysis, forecast) but rebuilds the delivery/storage layer around multiple users. Documented here so today's code is structured to make this jump easier later, without slowing down the personal tool.*

- **Design principle for now**: keep `fetch → store → analyze → forecast` logic decoupled from "who gets notified" (no hardcoded single recipient buried in analysis code) — this is the one thing worth doing today to make v2 painless.
- **Auth & user accounts**: Supabase Auth, Clerk, or Auth0 — don't hand-roll sign-up/login/password reset.
- **Data storage**: move from CSV to Postgres (Supabase's free tier includes this) — needed for concurrent multi-user reads/writes, user preferences, subscription status.
- **User preferences**: daily digest vs threshold alerts (e.g. "notify me when rate crosses X"), delivery time, unsubscribe.
- **Bulk email**: SendGrid, Resend, or Postmark — transactional email services built for deliverability at scale (personal SMTP won't scale or stay out of spam folders).
- **Payments** (if paid tiers): Stripe — subscriptions, billing, cancellations, webhooks for tier changes.
- **Hosting/frontend**: Streamlit isn't built for public auth + payments products. Likely move to FastAPI backend + a proper frontend (Next.js, or FastAPI + Jinja templates if staying Python-only). Streamlit could remain as an internal/admin dashboard.
- **Advertising**: straightforward technically (Google AdSense or direct sponsorships) but realistically needs meaningful traffic to matter — treat as a bonus revenue stream once there's an active user base, not a launch requirement.
- **Sequencing recommendation**: validate the personal tool works well and is genuinely useful first; treat the web product as a distinct build informed by that experience, not a parallel effort.

---

## 4. Key technical decisions locked in

| Decision | Choice | Reason |
|---|---|---|
| Scheduler | GitHub Actions cron | Free, reliable, doesn't depend on local machine |
| Primary data source | Banxico SIE API | Official rate, free, authoritative |
| Storage | CSV in-repo | Simplest, free, versioned, human-readable, sufficient scale (~8.7k rows, 1 append/day) |
| UI | Streamlit | Fastest path to a clean interactive dashboard in pure Python |
| Forecast model | LSTM/GRU (Keras) | Good fit for sequential data, manageable complexity |
| Validation | Walk-forward, not random split | Prevents lookahead leakage in time series |
| Email | SMTP via GitHub Secrets | Free, no new infra |
| Environment mgmt | venv + pip (or uv) | Matches GitHub Actions CI and deployment targets exactly; no conda needed since no complex binary/GPU deps |
| Python version | 3.11, installed via `uv` (isolated, not system-wide) | Fully supported by TensorFlow, Django, Streamlit, statsmodels; safely inside TF's 3.10-3.13 range without being on the bleeding edge where some packages lag. `uv python install 3.11` + `uv venv --python 3.11` keeps this project-local without touching the system Python |

---

## 5. Open items to confirm before/with Claude Code

- Exact email recipient(s) and preferred send time.
- Gmail app password vs SendGrid — pick one for `notifier.py`.
- Whether to seed `rates.csv` from your full historical TXT (recommended — gives the RNN 34 years of training data) or start fresh.
