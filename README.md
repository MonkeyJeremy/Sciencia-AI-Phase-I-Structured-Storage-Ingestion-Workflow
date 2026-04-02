# Sciencia AI — Phase I: Data Ingestion & Infrastructure

> **Pipeline stage:** Data Acquisition → Cleaning → Structured Storage → Splitting → Feature Engineering
> **Status:** Phase I complete · Steps 4 & 5 added

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
  [ Analysis ]  ──────────────────────────────  analyze_reviews.py
        │
        ▼
  [ Analyzer Dashboard ]  ────────────────────  app_review_analyzer.html
        │
        ▼
  [ Split ]  ─────────────────────────────────  split_dataset.py
        │         time-based, stratified
        ▼
   splits/  (train.csv / val.csv / test.csv)
        │
        ▼
  [ Feature Engineering ]  ───────────────────  feature_engineering.py
        │         TF-IDF, metadata, TextBlob,
        │         aspect keyword flags
        ▼
   outputs/features/  (model-ready matrices)
```

---

## Repository Structure

```
.
├── Google_scraper.py           # Step 1–2: Google Play review scraper & cleaning pipeline
├── db_ingestion.py             # Step 3:   CSV → SQLite ingestion workflow
├── analyze_reviews.py          # Step 3b:  Python analysis script — 4-page chart report
├── app_review_analyzer.html    #           Static dashboard for descriptive analysis
├── split_dataset.py            # Step 4:   Time-based stratified dataset splitting
├── feature_engineering.py      # Step 5:   TF-IDF, metadata, TextBlob & aspect features
├── requirements.txt            # Python dependencies
├── CONTRIBUTING.md             # Contribution guidelines
└── README.md
```

> An Apple App Store scraper (`apple_pipeline_1.py`) follows the same interface as `Google_scraper.py` and produces an `apple_reviews_cleaned.csv` compatible with `db_ingestion.py`.

---

## Example: ChatGPT (Google Play)

The pipeline was validated end-to-end using **ChatGPT** (`com.openai.chatgpt`) as the example app. Below is a summary of the results.

### Dataset at a glance

| Metric | Value |
|---|---|
| App | ChatGPT — Google Play (`com.openai.chatgpt`) |
| Raw reviews scraped | ~25,000 |
| After cleaning | **10,843** |
| Date range | 2026-02-21 → 2026-03-15 (23 days) |
| Unique users | 10,716 |
| Avg star rating | 4.46 ★ |
| Avg review length | 52 characters |
| Avg word count | 9.9 words |

### Sentiment distribution

| Sentiment | Count | % |
|---|---|---|
| Positive (4–5 ★) | 9,386 | 86.6% |
| Negative (1–2 ★) | 1,023 | 9.4% |
| Neutral (3 ★) | 434 | 4.0% |

### Key findings

- **Negative reviews are 2.3× longer** than positive ones (avg 103 ch vs 45 ch) — providing richer NLP signal per sample despite class imbalance.
- **Strong bimodal rating pattern** — 5★ (76.5%) and 1★ (7.4%) dominate; 2★ and 3★ are underrepresented.
- **41% of reviews are under 5 words** — emoji-only entries pass the minimum length filter but carry limited NLP signal. A ≥ 10-word threshold is recommended for the training set.
- **Kruskal-Wallis H = 500.13 (p < 0.001)** — review length differs significantly across star ratings.
- **Spearman ρ = −0.202** — higher-rated reviews tend to be shorter.
- **500-character truncation** — Google Play's hard character limit was hit by multiple reviewers; some feedback is incomplete.

### Ingestion run audit

| Field | Value |
|---|---|
| Run ID | 1 |
| Timestamp (UTC) | 2026-03-27 23:56:42 |
| Source file | `chatgpt_reviews_cleaned.csv` |
| Rows in file | 10,843 |
| Rows inserted | 10,843 |
| Rows skipped (duplicates) | 0 |
| Rows rejected | 0 |
| Status | SUCCESS |

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
python db_ingestion.py --source chatgpt_reviews_cleaned.csv --platform googleplay

# Both platforms at once
python db_ingestion.py \
    --source  googleplay_reviews_cleaned.csv --platform  googleplay \
    --source2 apple_reviews_cleaned.csv      --platform2 apple

# Dry-run (validate without writing)
python db_ingestion.py --source chatgpt_reviews_cleaned.csv --platform googleplay --dry-run

# Custom database path
python db_ingestion.py --source chatgpt_reviews_cleaned.csv --platform googleplay --db reviews_prod.db
```

### 4. Analyse the database

```bash
python analyze_reviews.py
# Outputs four chart pages:
#   page1_overview.png   — distributions & descriptive stats
#   page2_temporal.png   — monthly & daily trends
#   page3_stats.png      — statistical tests & correlations
#   page4_quality.png    — data quality audit
```

You can also query the database directly in Python:

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("reviews.db")

# Sentiment distribution
pd.read_sql("""
    SELECT sentiment, COUNT(*) AS n
    FROM sentiment_labels
    GROUP BY sentiment ORDER BY n DESC
""", conn)

# Avg star rating per month
pd.read_sql("""
    SELECT strftime('%Y-%m', review_date) AS month,
           ROUND(AVG(star_rating), 2)     AS avg_star,
           COUNT(*)                        AS n
    FROM reviews
    GROUP BY month ORDER BY month
""", conn)

conn.close()
```

### 5. Open the dashboard

Open `app_review_analyzer.html` in any modern browser — no server required. Enter an app name, select a platform, and click **Analyze**.

### 6. Split the dataset

```bash
# Default 70 / 15 / 15 split on all platforms
python split_dataset.py

# Filter to a single platform or app
python split_dataset.py --platform googleplay --app "ChatGPT"

# Custom ratios
python split_dataset.py --train 0.75 --val 0.10 --test 0.15

# Dry-run (print stats without writing files)
python split_dataset.py --dry-run
# Outputs:
#   splits/train.csv
#   splits/val.csv
#   splits/test.csv
```

> The splitter sorts by `review_date` before cutting, preventing temporal leakage. Within-split oversampling corrects the ~21.6× class imbalance identified in the EDA.

### 7. Build feature matrices

```bash
# Default: reads splits/ produced by split_dataset.py
python feature_engineering.py

# TF-IDF tuning
python feature_engineering.py --max-features 20000 --ngram-max 3

# Skip optional feature families
python feature_engineering.py --no-textblob
python feature_engineering.py --no-aspects

# Dry-run (validate inputs without writing features)
python feature_engineering.py --dry-run
# Outputs (in outputs/features/):
#   tfidf_{train,val,test}.npz       sparse TF-IDF matrices
#   meta_{train,val,test}.csv        numeric metadata features
#   textblob_{train,val,test}.csv    polarity & subjectivity scores
#   aspects_{train,val,test}.csv     aspect keyword flags
#   labels_{train,val,test}.csv      aligned sentiment labels
#   tfidf_vectorizer.pkl             fitted vectorizer (for inference)
#   feature_report.txt               human-readable validation report
```

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

| Metric | Target | ChatGPT result |
|---|---|---|
| Dataset size | ≥ 10,000 cleaned reviews | 10,843 ✓ |
| Retention rate | ≥ 80% of raw reviews | ~43% ⚠ (high non-English drop) |
| Null `star_rating` | < 5% | 0% ✓ |
| Duplicate review IDs | 0 | 0 ✓ |
| Class imbalance (Pos:Neu) | < 10× | ~21.6× ⚠ |
| Low-signal reviews (< 5 words) | < 20% | 41% ⚠ |

---

## Dependencies

See `requirements.txt`. Key packages:

| Package | Purpose |
|---|---|
| `google-play-scraper` | Google Play review API wrapper |
| `langdetect` | Language identification |
| `pandas` | Data manipulation |
| `matplotlib` / `seaborn` | Plotting (scraper diagnostics & analysis) |
| `scipy` | Statistical utilities & sparse matrix I/O |
| `scikit-learn` | TF-IDF vectorisation, metadata scaling (`split_dataset.py`, `feature_engineering.py`) |
| `textblob` | Rule-based polarity & subjectivity scores (`feature_engineering.py`) |

The dashboard (`app_review_analyzer.html`) is self-contained and requires no Python dependencies — Chart.js is loaded from CDN.

---

## Roadmap

- [ ] Apple App Store scraper (`apple_pipeline_1.py`)
- [x] Step 4: Time-based stratified dataset splitting (`split_dataset.py`)
- [x] Step 5: Feature engineering pipeline (`feature_engineering.py`)
- [ ] Scheduled runs (cron / Airflow DAG)
- [ ] PostgreSQL migration path
- [ ] Phase II: NLP labelling interface
- [ ] Phase III: Model training & evaluation loop

---

## License

Internal project — Sciencia AI. Not for public distribution.
