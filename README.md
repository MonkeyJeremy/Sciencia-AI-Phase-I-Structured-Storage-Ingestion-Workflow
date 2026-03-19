# Sciencia AI — Phase I: Structured Storage & Ingestion Workflow

## Overview

`db_ingestion.py` takes the **cleaned CSV** output from either pipeline and
loads it into a single normalised SQLite database (`reviews.db`).

---

## Why SQLite?

| Criterion | Decision |
|-----------|----------|
| Zero config | No server, credentials, or network — just a file |
| Single-file | `reviews.db` can be versioned, copied, or handed off directly |
| Full SQL | Supports all joins, aggregations, and window functions needed for labelling & training |
| Idempotent | Re-running ingestion is safe — duplicate `review_id`s are silently skipped |
| Upgrade path | Schema is forward-compatible with PostgreSQL/MySQL when the project scales |

---

## Schema

```
┌─────────────┐       ┌──────────────┐       ┌─────────────────┐
│  platforms  │       │     apps     │       │     users       │
│─────────────│       │──────────────│       │─────────────────│
│ id (PK)     │◄──┐   │ id (PK)      │       │ id (PK)         │
│ platform_key│   └───│ platform_fk  │   ┌───│ platform_fk     │
│ display_name│       │ app_name     │   │   │ username        │
└─────────────┘       └──────┬───────┘   │   └────────┬────────┘
                             │           │            │
                             ▼           │            ▼
                      ┌──────────────────────────────────────┐
                      │               reviews                │
                      │──────────────────────────────────────│
                      │ id (PK)                              │
                      │ review_id     TEXT UNIQUE  ← source  │
                      │ app_fk        → apps                 │
                      │ user_fk       → users                │
                      │ platform_fk   → platforms            │
                      │ star_rating   INTEGER 1–5            │
                      │ review_date   TEXT (YYYY-MM-DD)      │
                      │ review_text   TEXT                   │
                      │ review_length INTEGER                │
                      │ word_count    INTEGER                │
                      │ title         TEXT (Apple only)      │
                      │ ingestion_run → ingestion_runs       │
                      └──────────────┬───────────────────────┘
                                     │
                    ┌────────────────┴──────────────────┐
                    │                                   │
                    ▼                                   ▼
         ┌──────────────────────┐       ┌───────────────────────────┐
         │   sentiment_labels   │       │      ingestion_runs       │
         │──────────────────────│       │───────────────────────────│
         │ id (PK)              │       │ id (PK)                   │
         │ review_fk  UNIQUE    │       │ run_at                    │
         │ sentiment            │       │ source_file               │
         │ label_method         │       │ source_checksum (MD5)     │
         │ labelled_at          │       │ platform_key              │
         └──────────────────────┘       │ rows_in_file              │
                                        │ rows_inserted             │
                                        │ rows_skipped              │
                                        │ rows_rejected             │
                                        │ status                    │
                                        └───────────────────────────┘
```

### Design decisions

- **`sentiment_labels` is a separate table** — not a column on `reviews`.
  This lets downstream teams add new labelling methods (model-based, human
  labels, etc.) without touching the core fact table. The `label_method`
  column tracks which method produced each label.

- **`users` is scoped per platform** — a Google Play username and an Apple
  username are treated as distinct entities even if they're identical strings.

- **`ingestion_runs` as audit trail** — every execution writes one row with
  the source file path, MD5 checksum, and row counts. This makes the pipeline
  fully reproducible and debuggable.

- **`review_id` is UNIQUE** — re-running ingestion on the same CSV (or a
  CSV with overlapping rows) is completely safe. Duplicates are counted as
  `rows_skipped` and logged, but never raise an error.

---

## Install

```bash
pip install pandas
```

---

## Usage

```bash
# Google Play
python db_ingestion.py \
    --source googleplay_reviews_cleaned.csv \
    --platform googleplay

# Apple App Store
python db_ingestion.py \
    --source apple_reviews_cleaned.csv \
    --platform apple

# Both platforms into the same database
python db_ingestion.py \
    --source  googleplay_reviews_cleaned.csv --platform  googleplay \
    --source2 apple_reviews_cleaned.csv      --platform2 apple

# Custom database path
python db_ingestion.py \
    --source googleplay_reviews_cleaned.csv \
    --platform googleplay \
    --db /path/to/reviews_prod.db

# Dry-run — validate without writing
python db_ingestion.py \
    --source googleplay_reviews_cleaned.csv \
    --platform googleplay \
    --dry-run
```

---

## Outputs

| File | Description |
|------|-------------|
| `reviews.db` | SQLite database (default name, override with `--db`) |
| `ingestion.log` | Full run log with row counts and health checks |

---

## Connecting to the pipelines

```
googleplay_pipeline_1.py  ──►  googleplay_reviews_cleaned.csv  ──►  db_ingestion.py  ──►  reviews.db
apple_pipeline_1.py       ──►  apple_reviews_cleaned.csv        ──►  db_ingestion.py  ──►  reviews.db
```

Both pipelines write to the **same database**, so cross-platform queries work
out of the box:

```sql
-- Compare sentiment distribution across platforms
SELECT p.display_name, sl.sentiment, COUNT(*) AS n
FROM reviews r
JOIN platforms p ON r.platform_fk = p.id
JOIN sentiment_labels sl ON sl.review_fk = r.id
GROUP BY p.display_name, sl.sentiment
ORDER BY p.display_name, n DESC;

-- Average star rating per app across both stores
SELECT a.app_name, p.display_name, ROUND(AVG(r.star_rating), 2) AS avg_stars, COUNT(*) AS reviews
FROM reviews r
JOIN apps a ON r.app_fk = a.id
JOIN platforms p ON r.platform_fk = p.id
GROUP BY a.app_name, p.display_name
ORDER BY avg_stars DESC;

-- Audit: all ingestion runs
SELECT id, run_at, source_file, rows_inserted, rows_skipped, status
FROM ingestion_runs
ORDER BY run_at DESC;
```

---

## Full pipeline (end-to-end)

```bash
# Step 1 — Scrape & clean
python googleplay_pipeline_1.py
python apple_pipeline_1.py

# Step 2 — Load into structured database
python db_ingestion.py \
    --source  googleplay_reviews_cleaned.csv --platform  googleplay \
    --source2 apple_reviews_cleaned.csv      --platform2 apple

# Step 3 — Query
sqlite3 reviews.db "SELECT COUNT(*) FROM reviews;"
```
