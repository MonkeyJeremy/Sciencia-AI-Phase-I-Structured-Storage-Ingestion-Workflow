# Sciencia AI — Phase I: Data Ingestion & Infrastructure

> **Pipeline stage:** Data Acquisition → Cleaning → Structured Storage  
> **Status:** Phase I complete

---

## Overview

This repository contains the foundational data pipeline for Sciencia AI's sentiment analytics platform. It automates the collection of app store reviews, cleans and structures the raw data, and loads it into a normalised SQLite database — forming the entry point for downstream labelling, model training, and evaluation workflows.

```
Google Play / App Store
        │
        ▼
  [ Scraper ]  ──────────────────────────────  Google_scraper.py
        │                                       (+ Apple equivalent)
        ▼
  [ Cleaning ]  ─────────────────────────────  inline in scraper
        │         language detection, dedup,
        │         length filters, type coercion
        ▼
  [ DB Ingestion ]  ──────────────────────────  db_ingestion.py
        │
        ▼
   reviews.db  (SQLite)
        │
        ▼
  [ Analyzer Dashboard ]  ────────────────────  app_review_analyzer.html
```

---

## Repository Structure

```
.
├── Google_scraper.py           # Google Play review scraper & cleaning pipeline
├── db_ingestion.py             # CSV → SQLite ingestion workflow
├── app_review_analyzer.html    # Static dashboard for descriptive analysis
├── requirements.txt            # Python dependencies
├── CONTRIBUTING.md             # Contribution guidelines
└── README.md
```

> An Apple App Store scraper (`apple_pipeline_1.py`) follows the same interface as `Google_scraper.py` and produces an `apple_reviews_cleaned.csv` compatible with `db_ingestion.py`.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Scrape & clean reviews

```bash
python Google_scraper.py
# Outputs:
#   chatgpt_reviews_raw.csv
#   chatgpt_reviews_cleaned.csv
```

To change the target app, edit the constants near the top of the file:

```python
APP_ID   = 'com.openai.chatgpt'   # Google Play package ID
APP_NAME = 'ChatGPT'
TARGET   = 25_000                  # raw reviews to fetch before cleaning
```

### 3. Load into the database

```bash
# Single platform
python db_ingestion.py \
    --source chatgpt_reviews_cleaned.csv \
    --platform googleplay

# Both platforms at once
python db_ingestion.py \
    --source  googleplay_reviews_cleaned.csv --platform  googleplay \
    --source2 apple_reviews_cleaned.csv      --platform2 apple

# Dry-run (validate without writing)
python db_ingestion.py \
    --source chatgpt_reviews_cleaned.csv \
    --platform googleplay \
    --dry-run

# Custom database path
python db_ingestion.py \
    --source chatgpt_reviews_cleaned.csv \
    --platform googleplay \
    --db reviews_prod.db
```

### 4. Open the dashboard

Open `app_review_analyzer.html` in any modern browser — no server required. Enter an app name, select a platform, and click **Analyze**.

---

## Database Schema

The SQLite database (`reviews.db`) uses a normalised relational schema. All foreign keys are enforced and WAL mode is enabled for safe concurrent reads.

```
platforms ──┐
            ├── apps
            ├── users
            └── reviews ──── sentiment_labels
                    └─────── ingestion_runs
```

| Table | Purpose |
|---|---|
| `platforms` | One row per store (`googleplay`, `apple`) |
| `apps` | One row per (app, platform) pair |
| `users` | Normalised usernames, scoped per platform |
| `reviews` | Core fact table — idempotent on `review_id` |
| `sentiment_labels` | Derived labels, separated for easy re-labelling |
| `ingestion_runs` | Audit log — every run is traceable and replayable |

### Sentiment mapping (rule-based, `label_method = 'star_rule'`)

| Stars | Sentiment |
|---|---|
| 4 – 5 | `positive` |
| 3 | `neutral` |
| 1 – 2 | `negative` |
| null | `unrated` |

---

## Data Cleaning Steps

The scraper applies the following filters in order, logging rows dropped at each step:

| Step | Filter |
|---|---|
| `missing_date` | Drop rows with no review date |
| `null_empty_review` | Drop null or blank review text |
| `too_short` | Drop reviews under 3 characters |
| `non_english` | Drop non-English reviews (via `langdetect`) |
| `duplicate_id` | Deduplicate on `review_id` |

Derived columns added after cleaning: `review_length`, `word_count`, `sentiment`.

---

## Quality Targets

| Metric | Target |
|---|---|
| Dataset size | ≥ 10,000 cleaned reviews |
| Retention rate | ≥ 80% of raw reviews |
| Null `star_rating` | < 5% |
| Duplicate review IDs | 0 |
| Class imbalance (Pos:Neu) | < 10× |
| Low-signal reviews (< 5 words) | < 20% |

---

## Dependencies

See `requirements.txt`. Key packages:

| Package | Purpose |
|---|---|
| `google-play-scraper` | Google Play review API wrapper |
| `langdetect` | Language identification |
| `pandas` | Data manipulation |
| `matplotlib` / `seaborn` | Plotting (scraper diagnostics) |
| `scipy` | Statistical utilities |

The dashboard (`app_review_analyzer.html`) is self-contained and requires no Python dependencies — Chart.js is loaded from CDN.

---

## Roadmap

- [ ] Apple App Store scraper (`apple_pipeline_1.py`)
- [ ] Scheduled runs (cron / Airflow DAG)
- [ ] PostgreSQL migration path
- [ ] Phase II: NLP labelling interface
- [ ] Phase III: Model training & evaluation loop

---

## License

Internal project — Sciencia AI. Not for public distribution.
