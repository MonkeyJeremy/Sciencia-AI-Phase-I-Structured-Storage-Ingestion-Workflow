"""
split_dataset.py
===============================================================================
Sciencia AI — Phase I: Data Ingestion & Infrastructure
Step 4: Reproducible, Leakage-Free Dataset Splitting

PURPOSE
-------
Reads reviews from the SQLite database produced by db_ingestion.py and
exports three stratified, time-ordered CSV files ready for model training:

    outputs/
        train.csv   — majority of data, used to fit the model
        val.csv     — held-out slice for hyperparameter tuning
        test.csv    — final held-out slice, touched only at evaluation time

WHY TIME-BASED SPLITTING MATTERS
---------------------------------
Randomly shuffling reviews before splitting would cause temporal leakage:
the model would see future reviews during training and appear to generalise
well, but would fail in production where it only ever sees new reviews.

This script always sorts by review_date first, then cuts the timeline into
train / val / test bands. Reviews in the test set are always newer than any
review in the training set — exactly mirroring real deployment conditions.

STRATIFICATION
--------------
A naive time split can produce class-imbalanced splits (e.g. if the app had
a PR incident in the test window, negative reviews spike). After the time
cut, each split is rebalanced so that the positive : neutral : negative ratio
is consistent across all three sets — without shuffling rows across time
boundaries. This is done by up-sampling minority classes within each split
using random oversampling (with a fixed seed for reproducibility).

AUDIT TRAIL
-----------
Every run writes one row to the `dataset_splits` table in reviews.db,
recording the exact parameters, row counts, date boundaries, and a checksum
of each output file. This lets downstream teams pin their model training to a
specific split version and reproduce it exactly.

SCHEMA EXTENSION (applied automatically on first run)
------------------------------------------------------
    dataset_splits
        id, split_at, db_source, platform_filter, app_filter,
        cutoff_train, cutoff_val, label_method,
        n_train, n_val, n_test,
        train_file, val_file, test_file,
        train_md5,  val_md5,  test_md5,
        stratified, random_seed, notes

USAGE
-----
  # Default 70 / 15 / 15 split on all platforms
  python split_dataset.py

  # Custom ratios
  python split_dataset.py --train 0.75 --val 0.10 --test 0.15

  # Filter to a single platform or app
  python split_dataset.py --platform googleplay
  python split_dataset.py --app "ChatGPT"
  python split_dataset.py --platform googleplay --app "ChatGPT"

  # Disable stratification (keep raw time distribution)
  python split_dataset.py --no-stratify

  # Custom output directory and DB path
  python split_dataset.py --db reviews_prod.db --out-dir splits/v2

  # Dry-run: print stats without writing any files
  python split_dataset.py --dry-run

INSTALL
-------
  pip install pandas

===============================================================================
"""

import sqlite3
import hashlib
import logging
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level    = logging.INFO,
    format   = "[%(asctime)s] %(levelname)s  %(message)s",
    datefmt  = "%Y-%m-%d %H:%M:%S",
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("splitting.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# SCHEMA EXTENSION
# ─────────────────────────────────────────────
SPLIT_SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS dataset_splits (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,

    -- When and what
    split_at        TEXT    NOT NULL,           -- UTC timestamp of this run
    db_source       TEXT    NOT NULL,           -- path to reviews.db used
    platform_filter TEXT,                       -- NULL = all platforms
    app_filter      TEXT,                       -- NULL = all apps
    label_method    TEXT    NOT NULL DEFAULT 'star_rule',

    -- Time boundary parameters
    cutoff_train    TEXT    NOT NULL,           -- last review_date in train set
    cutoff_val      TEXT    NOT NULL,           -- last review_date in val set
    train_ratio     REAL    NOT NULL,
    val_ratio       REAL    NOT NULL,
    test_ratio      REAL    NOT NULL,

    -- Row counts (after stratification)
    n_total         INTEGER NOT NULL,
    n_train         INTEGER NOT NULL,
    n_val           INTEGER NOT NULL,
    n_test          INTEGER NOT NULL,

    -- Class distribution per split (JSON)
    dist_train      TEXT,                       -- {"positive":N,"neutral":N,"negative":N}
    dist_val        TEXT,
    dist_test       TEXT,

    -- Output file paths and checksums
    train_file      TEXT    NOT NULL,
    val_file        TEXT    NOT NULL,
    test_file       TEXT    NOT NULL,
    train_md5       TEXT,
    val_md5         TEXT,
    test_md5        TEXT,

    -- Reproducibility
    stratified      INTEGER NOT NULL DEFAULT 1, -- 1 = stratification applied
    random_seed     INTEGER NOT NULL,
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_splits_at ON dataset_splits(split_at);
"""


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def class_dist(df: pd.DataFrame) -> dict:
    """Return sentiment class counts as a plain dict."""
    counts = df["sentiment"].value_counts().to_dict()
    # Ensure all three keys always present even if a class has 0 rows
    for label in ("positive", "neutral", "negative", "unrated"):
        counts.setdefault(label, 0)
    return counts


def stratify_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Oversample minority classes within a split so every class has the same
    count as the majority class. Uses random sampling with replacement,
    seeded for reproducibility. Preserves the original rows — only adds
    duplicates of minority-class rows.

    This is done *within* each time-bounded split, not across splits, so
    no future data leaks into an earlier split.
    """
    max_count = df["sentiment"].value_counts().max()
    parts = []
    for label, group in df.groupby("sentiment"):
        if len(group) < max_count:
            extra = group.sample(
                n       = max_count - len(group),
                replace = True,
                random_state = seed,
            )
            parts.append(pd.concat([group, extra], ignore_index=True))
        else:
            parts.append(group)
    balanced = pd.concat(parts, ignore_index=True)
    # Shuffle so classes are interleaved (still seeded)
    return balanced.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_reviews(db_path: Path, platform: str | None, app: str | None) -> pd.DataFrame:
    """
    Pull all labelled reviews from the database, joining reviews →
    sentiment_labels → apps → platforms. Returns a DataFrame sorted
    by review_date ascending (oldest first).
    """
    conn = sqlite3.connect(db_path)

    query = """
        SELECT
            r.id            AS db_id,
            r.review_id,
            a.app_name,
            p.platform_key,
            r.star_rating,
            r.review_date,
            r.review_text,
            r.review_length,
            r.word_count,
            r.title,
            sl.sentiment,
            sl.label_method
        FROM reviews r
        JOIN sentiment_labels sl ON sl.review_fk = r.id
        JOIN apps      a ON r.app_fk      = a.id
        JOIN platforms p ON r.platform_fk = p.id
        WHERE r.review_date IS NOT NULL
    """

    params: list = []
    if platform:
        query  += " AND p.platform_key = ?"
        params.append(platform)
    if app:
        query  += " AND a.app_name = ?"
        params.append(app)

    query += " ORDER BY r.review_date ASC, r.id ASC"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def compute_cutoffs(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    """
    Given a DataFrame sorted by date, return the date strings that mark the
    end of the train window and the end of the val window.

    Cutting on *index position* (not calendar time) ensures the ratios hold
    even when review volume is unevenly distributed across months.
    """
    n           = len(df)
    train_end   = int(n * train_ratio)
    val_end     = int(n * (train_ratio + val_ratio))

    # Use the review_date of the last row in each window as the cutoff label
    cutoff_train = df.iloc[train_end - 1]["review_date"]
    cutoff_val   = df.iloc[val_end   - 1]["review_date"]
    return train_end, val_end, cutoff_train, cutoff_val


# ─────────────────────────────────────────────
# CORE SPLIT FUNCTION
# ─────────────────────────────────────────────

def split(
    db_path:      Path,
    out_dir:      Path,
    train_ratio:  float = 0.70,
    val_ratio:    float = 0.15,
    test_ratio:   float = 0.15,
    platform:     str | None = None,
    app:          str | None = None,
    label_method: str  = "star_rule",
    stratified:   bool = True,
    seed:         int  = 42,
    dry_run:      bool = False,
    notes:        str  = "",
) -> dict:

    # ── Validation ────────────────────────────────────────────────────────────
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0 — got {train_ratio}+{val_ratio}+{test_ratio}"
            f"={train_ratio+val_ratio+test_ratio:.4f}"
        )

    log.info("=" * 65)
    log.info(" SCIENCIA AI — DATASET SPLIT WORKFLOW")
    log.info(f" DB           : {db_path}")
    log.info(f" Output dir   : {out_dir}")
    log.info(f" Ratios       : train={train_ratio}  val={val_ratio}  test={test_ratio}")
    log.info(f" Platform     : {platform or 'all'}")
    log.info(f" App          : {app or 'all'}")
    log.info(f" Stratified   : {stratified}")
    log.info(f" Random seed  : {seed}")
    log.info(f" Dry-run      : {dry_run}")
    log.info("=" * 65)

    # ── Load ─────────────────────────────────────────────────────────────────
    log.info("[1/5] Loading reviews from database...")
    df = load_reviews(db_path, platform, app)

    if len(df) == 0:
        raise ValueError("No reviews found matching the given filters.")

    # Filter by label_method if desired
    df = df[df["label_method"] == label_method].reset_index(drop=True)

    log.info(f"      {len(df):,} labelled reviews loaded (method='{label_method}')")
    log.info(f"      Date range: {df['review_date'].min()}  →  {df['review_date'].max()}")

    dist_all = class_dist(df)
    log.info(f"      Overall class distribution: {dist_all}")

    if len(df) < 100:
        log.warning("      Very small dataset — splits may not be representative.")

    # ── Time-based cut ────────────────────────────────────────────────────────
    log.info("[2/5] Computing time-based split boundaries...")
    train_end, val_end, cutoff_train, cutoff_val = compute_cutoffs(
        df, train_ratio, val_ratio
    )

    df_train_raw = df.iloc[:train_end].copy()
    df_val_raw   = df.iloc[train_end:val_end].copy()
    df_test_raw  = df.iloc[val_end:].copy()

    log.info(f"      Train : rows 0–{train_end-1:,}  | cutoff date: {cutoff_train}")
    log.info(f"      Val   : rows {train_end:,}–{val_end-1:,}  | cutoff date: {cutoff_val}")
    log.info(f"      Test  : rows {val_end:,}–{len(df)-1:,}  | (remainder)")
    log.info(f"      Raw sizes — train: {len(df_train_raw):,}  val: {len(df_val_raw):,}  test: {len(df_test_raw):,}")

    # ── Stratification ────────────────────────────────────────────────────────
    log.info("[3/5] Applying stratification...")
    if stratified:
        df_train = stratify_split(df_train_raw, seed)
        df_val   = stratify_split(df_val_raw,   seed + 1)
        df_test  = stratify_split(df_test_raw,  seed + 2)
        log.info(f"      Stratified sizes — train: {len(df_train):,}  val: {len(df_val):,}  test: {len(df_test):,}")
    else:
        df_train = df_train_raw.copy()
        df_val   = df_val_raw.copy()
        df_test  = df_test_raw.copy()
        log.info("      Stratification skipped (--no-stratify).")

    dist_train = class_dist(df_train)
    dist_val   = class_dist(df_val)
    dist_test  = class_dist(df_test)

    log.info(f"      Train dist : {dist_train}")
    log.info(f"      Val   dist : {dist_val}")
    log.info(f"      Test  dist : {dist_test}")

    if dry_run:
        log.info("[DRY-RUN] No files written. Re-run without --dry-run to produce CSVs.")
        return {
            "n_total"       : len(df),
            "n_train"       : len(df_train),
            "n_val"         : len(df_val),
            "n_test"        : len(df_test),
            "cutoff_train"  : cutoff_train,
            "cutoff_val"    : cutoff_val,
            "dist_train"    : dist_train,
            "dist_val"      : dist_val,
            "dist_test"     : dist_test,
            "status"        : "dry_run",
        }

    # ── Write CSVs ────────────────────────────────────────────────────────────
    log.info("[4/5] Writing CSV files...")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Drop internal DB columns not needed downstream
    drop_cols = ["db_id", "label_method"]
    export_cols = [c for c in df_train.columns if c not in drop_cols]

    train_path = out_dir / "train.csv"
    val_path   = out_dir / "val.csv"
    test_path  = out_dir / "test.csv"

    df_train[export_cols].to_csv(train_path, index=False)
    df_val[export_cols].to_csv(val_path,     index=False)
    df_test[export_cols].to_csv(test_path,   index=False)

    train_md5 = md5_file(train_path)
    val_md5   = md5_file(val_path)
    test_md5  = md5_file(test_path)

    log.info(f"      train.csv  → {train_path}  ({len(df_train):,} rows)  MD5: {train_md5}")
    log.info(f"      val.csv    → {val_path}    ({len(df_val):,} rows)  MD5: {val_md5}")
    log.info(f"      test.csv   → {test_path}   ({len(df_test):,} rows)  MD5: {test_md5}")

    # ── Register split in DB ──────────────────────────────────────────────────
    log.info("[5/5] Recording split metadata to database...")
    conn = sqlite3.connect(db_path)
    conn.executescript(SPLIT_SCHEMA_SQL)
    conn.commit()

    split_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO dataset_splits (
            split_at, db_source, platform_filter, app_filter, label_method,
            cutoff_train, cutoff_val,
            train_ratio, val_ratio, test_ratio,
            n_total, n_train, n_val, n_test,
            dist_train, dist_val, dist_test,
            train_file, val_file, test_file,
            train_md5,  val_md5,  test_md5,
            stratified, random_seed, notes
        ) VALUES (
            ?,?,?,?,?,
            ?,?,
            ?,?,?,
            ?,?,?,?,
            ?,?,?,
            ?,?,?,
            ?,?,?,
            ?,?,?
        )
    """, (
        split_at, str(db_path), platform, app, label_method,
        cutoff_train, cutoff_val,
        train_ratio, val_ratio, test_ratio,
        len(df), len(df_train), len(df_val), len(df_test),
        json.dumps(dist_train), json.dumps(dist_val), json.dumps(dist_test),
        str(train_path), str(val_path), str(test_path),
        train_md5, val_md5, test_md5,
        int(stratified), seed, notes or None,
    ))
    split_id = cur.lastrowid
    conn.commit()
    conn.close()

    log.info(f"      Split ID {split_id} recorded in dataset_splits.")
    log.info("")
    log.info("  ── Summary ──────────────────────────────────────────")
    log.info(f"    Total reviews    : {len(df):,}")
    log.info(f"    Train            : {len(df_train):,}  ({len(df_train)/len(df)*100:.1f}%)")
    log.info(f"    Val              : {len(df_val):,}  ({len(df_val)/len(df)*100:.1f}%)")
    log.info(f"    Test             : {len(df_test):,}  ({len(df_test)/len(df)*100:.1f}%)")
    log.info(f"    Train cutoff     : {cutoff_train}")
    log.info(f"    Val cutoff       : {cutoff_val}")
    log.info(f"    Stratified       : {stratified}")
    log.info(f"    Random seed      : {seed}")
    log.info(f"    Split ID         : {split_id}")
    log.info(f"    Output dir       : {out_dir}/")
    log.info("=" * 65)

    return {
        "split_id"      : split_id,
        "n_total"       : len(df),
        "n_train"       : len(df_train),
        "n_val"         : len(df_val),
        "n_test"        : len(df_test),
        "cutoff_train"  : cutoff_train,
        "cutoff_val"    : cutoff_val,
        "dist_train"    : dist_train,
        "dist_val"      : dist_val,
        "dist_test"     : dist_test,
        "train_file"    : str(train_path),
        "val_file"      : str(val_path),
        "test_file"     : str(test_path),
        "train_md5"     : train_md5,
        "val_md5"       : val_md5,
        "test_md5"      : test_md5,
        "status"        : "success",
    }


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description     = "Sciencia AI — Time-based stratified dataset splitter",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog          = """
Examples:
  python split_dataset.py
  python split_dataset.py --train 0.75 --val 0.10 --test 0.15
  python split_dataset.py --platform googleplay
  python split_dataset.py --platform googleplay --app "ChatGPT"
  python split_dataset.py --no-stratify --seed 0
  python split_dataset.py --db reviews_prod.db --out-dir splits/v2
  python split_dataset.py --dry-run
        """,
    )
    parser.add_argument("--db",           default="reviews.db",
                        help="Path to SQLite database (default: reviews.db)")
    parser.add_argument("--out-dir",      default="splits",
                        help="Output directory for CSV files (default: splits/)")
    parser.add_argument("--train",        type=float, default=0.70,
                        help="Fraction for training set (default: 0.70)")
    parser.add_argument("--val",          type=float, default=0.15,
                        help="Fraction for validation set (default: 0.15)")
    parser.add_argument("--test",         type=float, default=0.15,
                        help="Fraction for test set (default: 0.15)")
    parser.add_argument("--platform",     default=None, choices=["googleplay", "apple"],
                        help="Filter to a single platform")
    parser.add_argument("--app",          default=None,
                        help="Filter to a single app by name (e.g. 'ChatGPT')")
    parser.add_argument("--label-method", default="star_rule",
                        help="Label method to filter on (default: star_rule)")
    parser.add_argument("--no-stratify",  action="store_true",
                        help="Disable within-split oversampling")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Random seed for oversampling (default: 42)")
    parser.add_argument("--notes",        default="",
                        help="Optional notes to record with this split")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Validate and print stats without writing files")

    # parse_known_args instead of parse_args so that Jupyter's kernel arguments
    # (e.g. -f kernel-xxx.json) are silently ignored rather than causing SystemExit
    args, _unknown = parser.parse_known_args()

    split(
        db_path      = Path(args.db),
        out_dir      = Path(args.out_dir),
        train_ratio  = args.train,
        val_ratio    = args.val,
        test_ratio   = args.test,
        platform     = args.platform,
        app          = args.app,
        label_method = args.label_method,
        stratified   = not args.no_stratify,
        seed         = args.seed,
        dry_run      = args.dry_run,
        notes        = args.notes,
    )


# ─────────────────────────────────────────────
# JUPYTER-FRIENDLY DIRECT CALL
# ─────────────────────────────────────────────
# If you are running this inside a Jupyter notebook, call split() directly
# instead of main() to avoid any argument-parsing entirely. Example:
#
#   from split_dataset import split
#   from pathlib import Path
#
#   result = split(
#       db_path     = Path("reviews.db"),
#       out_dir     = Path("splits"),
#       train_ratio = 0.70,
#       val_ratio   = 0.15,
#       test_ratio  = 0.15,
#       platform    = "googleplay",   # or None for all platforms
#       app         = "ChatGPT",      # or None for all apps
#       stratified  = True,
#       seed        = 42,
#       dry_run     = False,
#   )
#   print(result)

if __name__ == "__main__":
    main()
