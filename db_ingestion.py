"""
db_ingestion.py
===============================================================================
Sciencia AI — Phase I: Data Ingestion & Infrastructure
Step 3: Structured Storage & Reproducible Ingestion Workflow

PURPOSE
-------
Takes the cleaned CSV outputs from either pipeline:
    - googleplay_reviews_cleaned.csv   (from googleplay_pipeline_1.py)
    - apple_reviews_cleaned.csv        (from apple_pipeline_1.py)

...and loads them into a single normalised SQLite database:
    - reviews.db

The schema separates raw entity data (apps, users, reviews) from derived
analytical data (sentiment_labels, ingestion_runs), so downstream teams
can query either the raw signal or the processed labels independently.

DATABASE CHOICE: SQLite
-----------------------
SQLite is chosen for this phase because:
  - Zero-config: no server process, credentials, or network setup required
  - Single-file: reviews.db can be versioned, copied, or handed off directly
  - Full SQL: supports all the joins, aggregations, and window functions
    needed for Phase II labelling and model training queries
  - Portable: works identically on Mac, Windows, Linux — no environment gaps
  - Upgrade path: the schema is written to be forward-compatible with
    PostgreSQL or MySQL if the project scales to a hosted database later

USAGE
-----
  # Load a single platform's cleaned CSV
  python db_ingestion.py --source googleplay_reviews_cleaned.csv --platform googleplay

  # Load Apple cleaned CSV
  python db_ingestion.py --source apple_reviews_cleaned.csv --platform apple

  # Load both at once
  python db_ingestion.py \\
      --source googleplay_reviews_cleaned.csv --platform googleplay \\
      --source2 apple_reviews_cleaned.csv     --platform2 apple

  # Custom DB path
  python db_ingestion.py --source googleplay_reviews_cleaned.csv \\
                         --platform googleplay --db reviews_prod.db

  # Dry-run (validate without writing)
  python db_ingestion.py --source googleplay_reviews_cleaned.csv \\
                         --platform googleplay --dry-run

INSTALL
-------
  pip install pandas

===============================================================================
"""

import sqlite3
import pandas as pd
import argparse
import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level    = logging.INFO,
    format   = "[%(asctime)s] %(levelname)s  %(message)s",
    datefmt  = "%Y-%m-%d %H:%M:%S",
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ingestion.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────
SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ── platforms ────────────────────────────────────────────────────────────────
-- One row per store. Allows future cross-platform analysis without ambiguity.
CREATE TABLE IF NOT EXISTS platforms (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    platform_key  TEXT    NOT NULL UNIQUE,          -- 'googleplay' | 'apple'
    display_name  TEXT    NOT NULL                  -- 'Google Play' | 'Apple App Store'
);

-- ── apps ─────────────────────────────────────────────────────────────────────
-- One row per (app, platform) pair. The same logical app may exist on both
-- stores under different IDs, so (app_name, platform_fk) is the natural key.
CREATE TABLE IF NOT EXISTS apps (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    app_name      TEXT    NOT NULL,
    platform_fk   INTEGER NOT NULL REFERENCES platforms(id),
    UNIQUE (app_name, platform_fk)
);

-- ── users ────────────────────────────────────────────────────────────────────
-- Normalised to avoid storing the same username thousands of times.
-- Usernames are scoped per platform (a Google user ≠ an Apple user even if
-- they share a username).
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL,
    platform_fk   INTEGER NOT NULL REFERENCES platforms(id),
    UNIQUE (username, platform_fk)
);

-- ── reviews ──────────────────────────────────────────────────────────────────
-- Core fact table. review_id is the source-system identifier (Google Play
-- reviewId / Apple review id). It is UNIQUE so re-running ingestion is
-- idempotent — duplicate rows are silently skipped.
CREATE TABLE IF NOT EXISTS reviews (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id       TEXT    NOT NULL UNIQUE,        -- source-system ID
    app_fk          INTEGER NOT NULL REFERENCES apps(id),
    user_fk         INTEGER NOT NULL REFERENCES users(id),
    platform_fk     INTEGER NOT NULL REFERENCES platforms(id),
    star_rating     INTEGER CHECK (star_rating BETWEEN 1 AND 5),
    review_date     TEXT,                           -- YYYY-MM-DD
    review_text     TEXT    NOT NULL,
    review_length   INTEGER NOT NULL DEFAULT 0,     -- character count
    word_count      INTEGER NOT NULL DEFAULT 0,     -- word count
    title           TEXT,                           -- Apple only; NULL for Google Play
    ingestion_run   INTEGER REFERENCES ingestion_runs(id)
);

-- ── sentiment_labels ─────────────────────────────────────────────────────────
-- Separated from reviews so we can add new labelling methods (rule-based,
-- model-based, human-labelled) without altering the core fact table.
-- label_method: 'star_rule' = mapping 4-5★→positive, 3★→neutral, 1-2★→negative
CREATE TABLE IF NOT EXISTS sentiment_labels (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    review_fk       INTEGER NOT NULL UNIQUE REFERENCES reviews(id),
    sentiment       TEXT    NOT NULL CHECK (sentiment IN ('positive','neutral','negative','unrated')),
    label_method    TEXT    NOT NULL DEFAULT 'star_rule',
    labelled_at     TEXT    NOT NULL
);

-- ── ingestion_runs ───────────────────────────────────────────────────────────
-- Audit table. Every execution of this script writes one row here, so you
-- can always trace which rows came from which run, and replay or debug any run.
CREATE TABLE IF NOT EXISTS ingestion_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at          TEXT    NOT NULL,               -- UTC timestamp
    source_file     TEXT    NOT NULL,               -- path to the CSV that was loaded
    source_checksum TEXT    NOT NULL,               -- MD5 of the CSV for reproducibility
    platform_key    TEXT    NOT NULL,
    rows_in_file    INTEGER NOT NULL,
    rows_inserted   INTEGER NOT NULL DEFAULT 0,
    rows_skipped    INTEGER NOT NULL DEFAULT 0,     -- duplicates
    rows_rejected   INTEGER NOT NULL DEFAULT 0,     -- failed validation
    status          TEXT    NOT NULL DEFAULT 'running'
                    CHECK (status IN ('running','success','failed'))
);

-- ── indexes ──────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_reviews_app       ON reviews(app_fk);
CREATE INDEX IF NOT EXISTS idx_reviews_platform  ON reviews(platform_fk);
CREATE INDEX IF NOT EXISTS idx_reviews_date      ON reviews(review_date);
CREATE INDEX IF NOT EXISTS idx_reviews_star      ON reviews(star_rating);
CREATE INDEX IF NOT EXISTS idx_sentiment_sent    ON sentiment_labels(sentiment);
CREATE INDEX IF NOT EXISTS idx_sentiment_method  ON sentiment_labels(label_method);
"""


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
PLATFORM_MAP = {
    "googleplay": "Google Play",
    "apple":      "Apple App Store",
}

REQUIRED_COLS   = {"review_id", "app_name", "user", "star_rating", "date", "review", "review_length", "sentiment"}
OPTIONAL_COLS   = {"title", "word_count"}


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_or_create(cur: sqlite3.Cursor, table: str, where: dict, insert: dict = None) -> int:
    """
    Return the `id` of the row matching `where` in `table`.
    If it does not exist, insert a new row with columns from `insert` (falls
    back to `where` if `insert` is None) and return the new id.
    """
    cols   = list(where.keys())
    clause = " AND ".join(f"{c} = ?" for c in cols)
    row    = cur.execute(f"SELECT id FROM {table} WHERE {clause}", list(where.values())).fetchone()
    if row:
        return row[0]
    data = {**where, **(insert or {})}
    placeholders = ", ".join("?" for _ in data)
    cur.execute(
        f"INSERT OR IGNORE INTO {table} ({', '.join(data.keys())}) VALUES ({placeholders})",
        list(data.values())
    )
    # fetchone again in case of a race on UNIQUE (shouldn't happen in SQLite single-writer)
    return cur.execute(f"SELECT id FROM {table} WHERE {clause}", list(where.values())).fetchone()[0]


def validate_row(row: pd.Series) -> tuple[bool, str]:
    """Return (is_valid, reason). Rejects rows that would corrupt the schema."""
    if not row.get("review_id"):
        return False, "empty review_id"
    if not isinstance(row.get("review"), str) or not row["review"].strip():
        return False, "empty review text"
    rating = row.get("star_rating")
    if pd.notna(rating) and int(rating) not in (1, 2, 3, 4, 5):
        return False, f"star_rating out of range: {rating}"
    return True, ""


# ─────────────────────────────────────────────
# CORE INGESTION
# ─────────────────────────────────────────────
def ingest(
    source_path: Path,
    platform_key: str,
    db_path: Path,
    dry_run: bool = False,
) -> dict:
    """
    Load a cleaned CSV into the database.

    Returns a summary dict with keys:
        rows_in_file, rows_inserted, rows_skipped, rows_rejected, status
    """
    log.info("=" * 65)
    log.info(" SCIENCIA AI — DB INGESTION WORKFLOW")
    log.info(f" Source   : {source_path}")
    log.info(f" Platform : {platform_key}")
    log.info(f" Database : {db_path}")
    log.info(f" Dry-run  : {dry_run}")
    log.info("=" * 65)

    if platform_key not in PLATFORM_MAP:
        raise ValueError(f"Unknown platform '{platform_key}'. Choose: {list(PLATFORM_MAP)}")
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    # ── Load CSV ─────────────────────────────────────────────────────────────
    log.info("[1/5] Loading CSV...")
    df = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    rows_in_file = len(df)
    log.info(f"      {rows_in_file:,} rows loaded from {source_path.name}")

    # ── Column check ─────────────────────────────────────────────────────────
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    for col in OPTIONAL_COLS:
        if col not in df.columns:
            df[col] = None
            log.info(f"      Optional column '{col}' not in CSV — defaulting to NULL")

    # ── Type coercions ───────────────────────────────────────────────────────
    df["star_rating"]   = pd.to_numeric(df["star_rating"],   errors="coerce")
    df["review_length"] = pd.to_numeric(df["review_length"], errors="coerce").fillna(0).astype(int)
    df["word_count"]    = pd.to_numeric(df.get("word_count", 0), errors="coerce").fillna(0).astype(int)
    df["date"]          = pd.to_datetime(df["date"], errors="coerce")

    if dry_run:
        log.info("[DRY-RUN] Validation complete — no data written to database.")
        invalid = sum(1 for _, row in df.iterrows() if not validate_row(row)[0])
        log.info(f"[DRY-RUN] {rows_in_file - invalid:,} rows would be inserted, {invalid:,} would be rejected.")
        return {"rows_in_file": rows_in_file, "rows_inserted": 0,
                "rows_skipped": 0, "rows_rejected": invalid, "status": "dry_run"}

    # ── Connect & create schema ───────────────────────────────────────────────
    log.info("[2/5] Connecting to database and applying schema...")
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    log.info(f"      Schema OK — {db_path}")

    cur = conn.cursor()

    # ── Ingestion run record ──────────────────────────────────────────────────
    log.info("[3/5] Registering ingestion run...")
    run_at   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    checksum = md5_file(source_path)
    cur.execute("""
        INSERT INTO ingestion_runs
            (run_at, source_file, source_checksum, platform_key, rows_in_file, status)
        VALUES (?, ?, ?, ?, ?, 'running')
    """, (run_at, str(source_path), checksum, platform_key, rows_in_file))
    run_id = cur.lastrowid
    conn.commit()
    log.info(f"      Run ID: {run_id}  |  Source MD5: {checksum}")

    # ── Ensure platform row exists ────────────────────────────────────────────
    platform_fk = get_or_create(
        cur, "platforms",
        where  = {"platform_key": platform_key},
        insert = {"display_name": PLATFORM_MAP[platform_key]},
    )

    # ── Row-by-row ingestion ──────────────────────────────────────────────────
    log.info("[4/5] Ingesting rows...")
    inserted = skipped = rejected = 0
    COMMIT_EVERY = 2_000

    for i, (_, row) in enumerate(df.iterrows()):

        valid, reason = validate_row(row)
        if not valid:
            log.debug(f"  REJECT row {i}: {reason}")
            rejected += 1
            continue

        review_id = str(row["review_id"]).strip()
        app_name  = str(row["app_name"]).strip()
        username  = (str(row.get("user") or "anonymous")).strip() or "anonymous"
        sentiment = str(row.get("sentiment") or "unrated").strip()
        title     = str(row["title"]).strip() if row.get("title") else None
        review_date = str(row["date"].date()) if pd.notna(row["date"]) else None
        star_rating = None if pd.isna(row["star_rating"]) else int(row["star_rating"])

        # Look-ups / inserts for normalised dimension tables
        app_fk  = get_or_create(cur, "apps",  {"app_name": app_name, "platform_fk": platform_fk})
        user_fk = get_or_create(cur, "users", {"username": username,  "platform_fk": platform_fk})

        # Insert review — IGNORE if review_id already exists (idempotency)
        cur.execute("""
            INSERT OR IGNORE INTO reviews
                (review_id, app_fk, user_fk, platform_fk, star_rating,
                 review_date, review_text, review_length, word_count, title, ingestion_run)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            review_id, app_fk, user_fk, platform_fk, star_rating,
            review_date, str(row["review"]), int(row["review_length"]),
            int(row.get("word_count") or 0), title, run_id,
        ))

        if cur.rowcount == 1:
            # New review inserted — also write its sentiment label
            review_fk = cur.lastrowid
            cur.execute("""
                INSERT OR IGNORE INTO sentiment_labels
                    (review_fk, sentiment, label_method, labelled_at)
                VALUES (?, ?, 'star_rule', ?)
            """, (review_fk, sentiment, run_at))
            inserted += 1
        else:
            skipped += 1

        if (i + 1) % COMMIT_EVERY == 0:
            conn.commit()
            log.info(f"      ... {i+1:>7,} rows processed  "
                     f"(inserted: {inserted:,}  skipped: {skipped:,}  rejected: {rejected:,})")

    conn.commit()

    # ── Update ingestion run record ───────────────────────────────────────────
    cur.execute("""
        UPDATE ingestion_runs
        SET rows_inserted = ?, rows_skipped = ?, rows_rejected = ?, status = 'success'
        WHERE id = ?
    """, (inserted, skipped, rejected, run_id))
    conn.commit()

    # ── Health check ─────────────────────────────────────────────────────────
    log.info("[5/5] Running health checks...")

    def q(sql):
        return conn.execute(sql).fetchall()

    log.info("")
    log.info("  ── Database totals ──────────────────────────────")
    log.info(f"  Total reviews      : {q('SELECT COUNT(*) FROM reviews')[0][0]:>8,}")
    log.info(f"  Unique apps        : {q('SELECT COUNT(*) FROM apps')[0][0]:>8,}")
    log.info(f"  Unique users       : {q('SELECT COUNT(*) FROM users')[0][0]:>8,}")
    log.info(f"  Sentiment labels   : {q('SELECT COUNT(*) FROM sentiment_labels')[0][0]:>8,}")
    log.info("")

    log.info("  ── Sentiment distribution ───────────────────────")
    for label, count in q("""
        SELECT sl.sentiment, COUNT(*) AS n
        FROM sentiment_labels sl
        JOIN reviews r ON sl.review_fk = r.id
        JOIN platforms p ON r.platform_fk = p.id
        WHERE p.platform_key = ?
        GROUP BY sl.sentiment ORDER BY n DESC
    """, ) if False else conn.execute("""
        SELECT sl.sentiment, COUNT(*) AS n
        FROM sentiment_labels sl
        JOIN reviews r ON sl.review_fk = r.id
        JOIN platforms p ON r.platform_fk = p.id
        WHERE p.platform_key = ?
        GROUP BY sl.sentiment ORDER BY n DESC
    """, (platform_key,)).fetchall():
        log.info(f"    {label:<12}: {count:>7,}")

    log.info("")
    log.info("  ── Star rating distribution ─────────────────────")
    for star, count in conn.execute("""
        SELECT r.star_rating, COUNT(*) AS n
        FROM reviews r
        JOIN platforms p ON r.platform_fk = p.id
        WHERE p.platform_key = ? AND r.star_rating IS NOT NULL
        GROUP BY r.star_rating ORDER BY r.star_rating
    """, (platform_key,)).fetchall():
        log.info(f"    {star}★  : {count:>7,}")

    log.info("")
    log.info("  ── Top apps by review count ─────────────────────")
    for app, count in conn.execute("""
        SELECT a.app_name, COUNT(*) AS n
        FROM reviews r
        JOIN apps a ON r.app_fk = a.id
        JOIN platforms p ON r.platform_fk = p.id
        WHERE p.platform_key = ?
        GROUP BY a.app_name ORDER BY n DESC LIMIT 10
    """, (platform_key,)).fetchall():
        log.info(f"    {app:<28}: {count:>6,}")

    log.info("")
    log.info("  ── Ingestion run summary ────────────────────────")
    log.info(f"    Run ID            : {run_id}")
    log.info(f"    Source file       : {source_path.name}")
    log.info(f"    Source MD5        : {checksum}")
    log.info(f"    Rows in file      : {rows_in_file:,}")
    log.info(f"    Rows inserted     : {inserted:,}")
    log.info(f"    Rows skipped      : {skipped:,}  (duplicates — idempotent)")
    log.info(f"    Rows rejected     : {rejected:,}  (failed validation)")
    log.info(f"    Status            : SUCCESS")
    log.info("")

    conn.close()
    log.info(f"Database saved → {db_path}")
    log.info("=" * 65)

    return {
        "run_id"        : run_id,
        "rows_in_file"  : rows_in_file,
        "rows_inserted" : inserted,
        "rows_skipped"  : skipped,
        "rows_rejected" : rejected,
        "status"        : "success",
    }


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description = "Sciencia AI — Load cleaned review CSV into SQLite",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
Examples:
  python db_ingestion.py --source googleplay_reviews_cleaned.csv --platform googleplay
  python db_ingestion.py --source apple_reviews_cleaned.csv      --platform apple
  python db_ingestion.py --source googleplay_reviews_cleaned.csv --platform googleplay --dry-run
  python db_ingestion.py \\
      --source  googleplay_reviews_cleaned.csv --platform  googleplay \\
      --source2 apple_reviews_cleaned.csv      --platform2 apple
        """,
    )
    parser.add_argument("--source",    required=True,  help="Path to cleaned CSV (required)")
    parser.add_argument("--platform",  required=True,  choices=["googleplay","apple"],
                        help="Platform key: 'googleplay' or 'apple'")
    parser.add_argument("--source2",   default=None,   help="Optional second cleaned CSV")
    parser.add_argument("--platform2", default=None,   choices=["googleplay","apple"],
                        help="Platform key for second CSV")
    parser.add_argument("--db",        default="reviews.db",
                        help="Path to SQLite database file (default: reviews.db)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Validate CSV without writing to the database")
    args = parser.parse_args()

    db_path = Path(args.db)

    # First source
    result1 = ingest(
        source_path  = Path(args.source),
        platform_key = args.platform,
        db_path      = db_path,
        dry_run      = args.dry_run,
    )

    # Optional second source
    if args.source2:
        if not args.platform2:
            parser.error("--platform2 is required when --source2 is provided")
        result2 = ingest(
            source_path  = Path(args.source2),
            platform_key = args.platform2,
            db_path      = db_path,
            dry_run      = args.dry_run,
        )
        log.info("Both sources ingested successfully.")
        log.info(f"  Source 1 inserted: {result1['rows_inserted']:,}")
        log.info(f"  Source 2 inserted: {result2['rows_inserted']:,}")


if __name__ == "__main__":
    main()
