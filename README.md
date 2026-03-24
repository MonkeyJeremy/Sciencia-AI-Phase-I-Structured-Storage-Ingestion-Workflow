# 📱 App Store Review Data Pipeline
### Sciencia AI Internship · Phase I: Data Ingestion & Infrastructure

A production-grade data pipeline that scrapes, cleans, and stores app reviews from the **Google Play Store** and **Apple App Store** — feeding a downstream NLP/sentiment analysis system. Built as part of a structured internship program at Sciencia AI.

---

## 🗂 Repository Structure

```
app-review-pipeline/
│
├── apple_pipeline_1.py          # Apple App Store scraper & ETL pipeline
├── googleplay_pipeline_1.py     # Google Play scraper & ETL pipeline
├── Google_scraper.py            # Standalone deep-scrape (e.g. ChatGPT, 25k reviews)
├── app_review_analyzer.html     # Interactive review analytics dashboard (browser)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│                                                                 │
│   Apple App Store API          Google Play Internal API         │
│   (4 countries × 500 cap)      (continuation token pagination)  │
└──────────────┬──────────────────────────┬───────────────────────┘
               │                          │
               ▼                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     STEP 1 — SCRAPE                             │
│                                                                 │
│  • apple_pipeline_1.py     scrapes US / GB / AU / CA           │
│  • googleplay_pipeline_1.py  paginates with Sort.NEWEST         │
│  • Deduplication by review_id across countries/batches          │
│  • Target: 2,000 reviews per app (10 apps = ~20,000 raw)        │
│                                                                 │
│  Output: apple_reviews_raw.csv / googleplay_reviews_raw.csv     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     STEP 2 — CLEAN                              │
│                                                                 │
│  Drop missing dates          Drop null/empty reviews            │
│  Drop < 3 char reviews       Drop non-English (langdetect)      │
│  Drop duplicate review IDs                                      │
│                                                                 │
│  Derive:  review_length  |  sentiment  (from star_rating)       │
│           positive ≥ 4★  |  negative ≤ 2★  |  neutral = 3★     │
│                                                                 │
│  Output: apple_reviews_cleaned.csv / googleplay_reviews_cleaned │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     STEP 3 — LOAD (SQLite)                      │
│                                                                 │
│  Normalised 3-table schema: apps · users · reviews              │
│  WAL journal mode for write performance                         │
│  INSERT OR IGNORE — safe to re-run (idempotent)                 │
│  Indexed on: sentiment, star_rating, date, app_fk               │
│                                                                 │
│  Output: apple_reviews.db / googleplay_reviews.db               │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     STEP 4 — HEALTH CHECKS                      │
│                                                                 │
│  Total reviews · Unique apps/users · Sentiment distribution     │
│  Star rating distribution · Avg review length · Reviews by year │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🗃 Database Schema

Both pipelines write to a normalised **SQLite** database with three tables.

```sql
-- One row per unique app
CREATE TABLE apps (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    app_name  TEXT NOT NULL UNIQUE
);

-- One row per unique username (anonymous if blank)
CREATE TABLE users (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    username  TEXT NOT NULL UNIQUE
);

-- Core review data, foreign-keyed to apps + users
CREATE TABLE reviews (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id      TEXT    UNIQUE,           -- platform-native ID (dedup key)
    app_fk         INTEGER NOT NULL REFERENCES apps(id),
    user_fk        INTEGER NOT NULL REFERENCES users(id),
    title          TEXT,                     -- Apple only (Play has no title)
    star_rating    INTEGER,                  -- 1–5
    date           TEXT,                     -- YYYY-MM-DD
    review         TEXT,
    sentiment      TEXT CHECK(sentiment IN ('positive','neutral','negative','unrated')),
    review_length  INTEGER DEFAULT 0,
    scraped_at     TEXT                      -- UTC timestamp
);

-- Indexes for fast downstream queries
CREATE INDEX idx_sentiment   ON reviews(sentiment);
CREATE INDEX idx_star_rating ON reviews(star_rating);
CREATE INDEX idx_date        ON reviews(date);
CREATE INDEX idx_app         ON reviews(app_fk);
```

**Sentiment mapping** (derived at clean time, never re-computed):

| star_rating | sentiment  |
|:-----------:|:----------:|
| ≥ 4         | positive   |
| 3           | neutral    |
| ≤ 2         | negative   |
| NULL        | unrated    |

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Google Play pipeline

```bash
python googleplay_pipeline_1.py
# Outputs: googleplay_reviews_raw.csv
#          googleplay_reviews_cleaned.csv
#          googleplay_reviews.db
```

### 3. Run the Apple App Store pipeline

```bash
python apple_pipeline_1.py
# Outputs: apple_reviews_raw.csv
#          apple_reviews_cleaned.csv
#          apple_reviews.db
```

### 4. Deep-scrape a single app (e.g. ChatGPT — 25k reviews)

```bash
python Google_scraper.py
# Outputs: chatgpt_reviews_raw.csv
#          chatgpt_reviews_cleaned.csv
```

### 5. Open the analytics dashboard

Just open `app_review_analyzer.html` in any browser — no server needed.

---

## 🔧 Configuration

Both pipelines expose a simple config block at the top of each file.

**Google Play** (`googleplay_pipeline_1.py`):
```python
TARGET_PER_APP   = 2_000   # reviews to collect per app
MIN_REVIEW_CHARS = 3       # minimum review length to keep
DELAY_SEC        = 0.5     # pause between API batches

APPS = [
    ("com.spotify.music", "Spotify"),
    ("com.openai.chatgpt", "ChatGPT"),
    # add / remove as needed
    # App ID = package name from play.google.com/store/apps/details?id=<APP_ID>
]
```

**Apple App Store** (`apple_pipeline_1.py`):
```python
TARGET_PER_APP = 2_000
COUNTRIES      = ["us", "gb", "au", "ca"]   # 4 × 500 cap = 2,000 max

APPS = [
    (324684580, "spotify", "Spotify"),
    # (app_id_int, slug, display_name)
    # App ID from: apps.apple.com/us/app/<slug>/id<APP_ID>
]
```

---

## 📊 Apps Scraped (Phase I)

| App            | Google Play ID                    | Apple App Store ID |
|----------------|-----------------------------------|--------------------|
| Spotify        | com.spotify.music                 | 324684580          |
| Instagram      | com.instagram.android             | 389801252          |
| WhatsApp       | com.whatsapp                      | 310633997          |
| Netflix        | com.netflix.mediaclient           | 363590051          |
| YouTube        | com.google.android.youtube        | 544007664          |
| Duolingo       | com.duolingo                      | 570060128          |
| Uber           | com.ubercab                       | 368677368          |
| Airbnb         | com.airbnb.android                | 401626263          |
| X (Twitter)    | com.twitter.android               | 333903271          |
| Reddit         | com.reddit.frontpage              | 1064216828         |

---

## 🧹 Cleaning Pipeline — Drop Logic

| Step                | Condition                              | Typical Drop Rate |
|---------------------|----------------------------------------|-------------------|
| Missing date        | `date IS NULL`                         | < 1%              |
| Null/empty review   | `review IS NULL OR review = ''`        | 1–3%              |
| Too short           | `len(review) < 3`                      | 1–2%              |
| Non-English         | `langdetect(review) != 'en'`           | 30–50%            |
| Duplicate review_id | keep first occurrence                  | < 1%              |

> **Note:** Non-English is the dominant drop. Google Play's `lang='en'` filter is advisory only — multilingual users still leave reviews in their native language. The Apple pipeline scrapes 4 English-speaking countries specifically to minimise this.

---

## 📈 Analytics Dashboard

`app_review_analyzer.html` is a self-contained, zero-dependency browser dashboard. Open it locally — no backend or API key needed.

**Features:**
- Platform toggle (Google Play / Apple App Store)
- App search with autocomplete (20 apps per platform)
- Star rating & sentiment distribution charts
- Monthly review volume with sentiment stacking
- Review length analysis by star rating and sentiment
- Cleaning funnel visualisation
- Pipeline readiness scorecard (7 automated checks)
- Actionable recommendations (class imbalance, NLP signal quality, etc.)

---

## 🗺 Project Roadmap

- [x] **Phase I** — Data ingestion: scraping, cleaning, SQLite storage
- [ ] **Phase II** — NLP: topic modelling, aspect-based sentiment analysis
- [ ] **Phase III** — Model training: fine-tuned classifier on review corpus
- [ ] **Phase IV** — Dashboard: live pipeline + model inference UI

---

## 🛠 Tech Stack

| Layer       | Technology                                              |
|-------------|----------------------------------------------------------|
| Scraping    | `google-play-scraper`, `apple-app-reviews-scraper`       |
| Processing  | `pandas`, `langdetect`                                   |
| Storage     | `sqlite3` (WAL mode, normalised schema)                  |
| Visualisation | `Chart.js 4.4`, vanilla HTML/CSS/JS                    |
| Analysis    | `matplotlib`, `seaborn`, `scipy`                         |

---

*Sciencia AI · Phase I: Data Ingestion & Infrastructure*
