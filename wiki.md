# Wiki — App Store Review Data Pipeline

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Why Two Pipelines?](#2-why-two-pipelines)
3. [How the Apple Pipeline Overcomes the 500-Review Cap](#3-how-the-apple-pipeline-overcomes-the-500-review-cap)
4. [How the Google Play Pipeline Handles Infinite Loops](#4-how-the-google-play-pipeline-handles-infinite-loops)
5. [Cleaning Decisions — Rationale](#5-cleaning-decisions--rationale)
6. [Sentiment Labelling Strategy](#6-sentiment-labelling-strategy)
7. [Database Design Decisions](#7-database-design-decisions)
8. [Analytics Dashboard](#8-analytics-dashboard)
9. [Known Limitations & Future Work](#9-known-limitations--future-work)
10. [Glossary](#10-glossary)

---

## 1. Project Overview

This project is **Phase I** of a multi-phase internship at Sciencia AI, focused on building infrastructure to collect and process app reviews at scale for downstream NLP and machine learning tasks.

**Goal:** Collect ≥ 2,000 English-language reviews per app per platform, clean them into a consistent schema, and load them into a queryable SQLite database — all in a single reproducible pipeline run.

**Phase I deliverables:**

| Deliverable | Description |
|---|---|
| `googleplay_pipeline_1.py` | Full ETL for Google Play (10 apps, ~20k reviews) |
| `apple_pipeline_1.py` | Full ETL for Apple App Store (10 apps, ~20k reviews) |
| `Google_scraper.py` | Standalone deep-scraper for a single app (25k target) |
| `app_review_analyzer.html` | Browser-based analytics dashboard |

---

## 2. Why Two Pipelines?

The two platforms have fundamentally different APIs and constraints:

| Concern | Google Play | Apple App Store |
|---|---|---|
| API type | Internal JSON endpoint (wrapped by `google-play-scraper`) | RSS/JSON API (wrapped by `apple-app-reviews-scraper`) |
| Auth | None required | Bearer token per app per country |
| Review cap | No hard cap — paginated via continuation token | **Hard cap: 500 reviews per app per country** |
| Language filter | `lang='en'` param (advisory — not reliable) | Must scrape English-speaking countries manually |
| Termination signal | No new review IDs in batch | `offset` returns `None` |

Because the constraints differ so much, two separate pipelines produce cleaner, more maintainable code than a single abstracted one.

---

## 3. How the Apple Pipeline Overcomes the 500-Review Cap

Apple's API caps reviews at **500 per app per country**. To reach the 2,000-review target, the pipeline scrapes from four English-speaking countries and deduplicates by `review_id`:

```
US store  → up to 500 unique reviews
GB store  → up to 500 unique reviews  (deduplicated against US)
AU store  → up to 500 unique reviews  (deduplicated against US+GB)
CA store  → up to 500 unique reviews  (deduplicated against US+GB+AU)
─────────────────────────────────────────────────────────────────
Maximum unique reviews: 2,000
```

The deduplication uses a `seen_ids` set that persists **across countries** for the same app. If the target is already met after the US store, remaining countries are skipped entirely.

---

## 4. How the Google Play Pipeline Handles Infinite Loops

A known issue with `google-play-scraper` is that the API returns a `continuation_token` **even after all reviews have been exhausted** — meaning a naive `while continuation_token:` loop will run forever.

The correct termination condition (per library docs) is: **stop when no new review IDs appear in a batch.**

```python
new_ids = {r['reviewId'] for r in result} - seen_ids
if not new_ids:
    print('No new review IDs in batch — all reviews collected.')
    break
```

This is implemented in both `googleplay_pipeline_1.py` and `Google_scraper.py`.

---

## 5. Cleaning Decisions — Rationale

Each cleaning step has a deliberate reason:

| Step | Why |
|---|---|
| Drop missing `date` | A review with no date can't be used for temporal analysis or drift detection |
| Drop null/empty `review` | Rating-only entries carry no NLP signal |
| Drop `len(review) < 3` | Catches single-emoji or punctuation-only reviews that slip through the empty check |
| Drop non-English | The downstream model is English-only; non-English text degrades tokeniser performance |
| Drop duplicate `review_id` | Prevents data leakage between train/test splits if the same review is seen twice |

**Why `langdetect` and not the API's `lang` param?**  
Google Play's `lang='en'` parameter is advisory. Users who write in French, Spanish, or Arabic still appear in the English-language result set because their account locale is `en`. `langdetect` catches these at clean time. The Apple pipeline mitigates the problem upstream by only scraping English-speaking country stores.

---

## 6. Sentiment Labelling Strategy

Sentiment is derived deterministically from `star_rating` — no model inference needed at this stage. This is intentional: Phase I is about infrastructure, not predictions.

```
star_rating ≥ 4  →  positive
star_rating = 3  →  neutral
star_rating ≤ 2  →  negative
star_rating NULL →  unrated
```

**Trade-offs of this approach:**

- ✅ Fast, reproducible, zero error rate
- ✅ Sufficient for class-balanced dataset construction in Phase II
- ⚠️ Misses sarcasm ("5 stars, love crashing every 30 seconds")
- ⚠️ 3★ reviews are genuinely ambiguous — some are mildly positive, some mildly negative

Phase II will train a text-based classifier to refine these labels.

---

## 7. Database Design Decisions

**Why SQLite and not PostgreSQL/Parquet?**  
SQLite is zero-infrastructure, file-portable, and sufficient for < 100k rows. The schema is normalised and indexed identically to how it would be in PostgreSQL — migrating later is a one-line change to the connection string.

**Why normalise into three tables?**  
Separating `apps` and `users` from `reviews`:
- Saves storage (app name stored once, not 2,000 times per app)
- Enables fast `GROUP BY app_fk` aggregations without string comparisons
- Makes it straightforward to add app metadata (category, developer) in a future column

**Why WAL journal mode?**  
WAL (Write-Ahead Logging) allows concurrent reads during a write. Even in a single-process pipeline, it significantly improves write throughput for bulk inserts.

**Why `INSERT OR IGNORE`?**  
The pipeline is designed to be **idempotent** — running it twice on the same data won't duplicate rows. `review_id` has a `UNIQUE` constraint, so duplicate inserts are silently skipped.

---

## 8. Analytics Dashboard

`app_review_analyzer.html` is a fully self-contained analytics tool — open it in any browser, no server or API key required.

**How it works:**  
The dashboard generates statistically-shaped synthetic data seeded by the app ID. This means the same app always produces the same dashboard, and different apps produce meaningfully different results — without requiring a live database connection.

**Pipeline readiness scorecard — checks performed:**

| Check | Pass threshold |
|---|---|
| Dataset size | ≥ 10,000 reviews |
| Retention rate | ≥ 80% of raw scraped |
| Null `star_rating` | < 5% |
| Duplicate `review_id` | 0 |
| Class imbalance (Pos:Neu) | < 10× |
| Low-signal reviews (< 5 words) | < 20% |
| Bimodality (1★ + 5★ share) | flagged if > 70% |

**Recommendation engine:**  
The dashboard automatically surfaces targeted recommendations when checks fail — e.g. multi-country scraping for low retention, SMOTE oversampling for severe class imbalance, word-count filtering for low NLP signal.

---

## 9. Known Limitations & Future Work

| Limitation | Planned Fix |
|---|---|
| Apple hard cap of 500/country | Add more countries (nz, ie, sg) in Phase II |
| `langdetect` misclassifies short reviews | Add a minimum word count (≥ 5 words) before language detection |
| Sentiment derived from star rating only | Phase II: train text classifier to refine labels |
| SQLite not suitable for concurrent writes | Phase III: migrate to PostgreSQL |
| No incremental scraping (always full re-scrape) | Add `scraped_after` date filter once schema is stable |
| `Google_scraper.py` not integrated into main pipeline | Merge into `googleplay_pipeline_1.py` as a `--deep` flag |

---

## 10. Glossary

| Term | Definition |
|---|---|
| `review_id` | Platform-native unique identifier for a review. Used as the deduplication key across all pipeline stages. |
| `continuation_token` | Google Play's pagination cursor. Returned with every batch; signals end-of-data only indirectly (via empty new-ID set). |
| `sentiment` | Derived label: positive / neutral / negative / unrated. Computed from `star_rating` at clean time. |
| `review_length` | Character count of the cleaned `review` string. |
| `app_fk` / `user_fk` | Foreign keys into the `apps` and `users` tables respectively. |
| WAL | Write-Ahead Logging — SQLite journal mode for improved write throughput. |
| Bimodality | A rating distribution strongly skewed toward both 1★ and 5★, with few 2–4★ reviews. Common in consumer apps. |
| Retention rate | `len(cleaned) / len(raw) × 100`. Target ≥ 80%. |
| NLP signal | Whether a review contains enough text for a model to learn from. Reviews under ~5 words are considered low-signal. |

---

*Sciencia AI · Phase I: Data Ingestion & Infrastructure*
