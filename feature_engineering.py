"""
feature_engineering.py
===============================================================================
Sciencia AI — Phase I: Data Ingestion & Infrastructure
Step 5: Feature Engineering Pipeline

PURPOSE
-------
Transforms cleaned review text from the splits produced by split_dataset.py
into model-ready feature matrices. Reads train/val/test CSVs, fits all
transformers on the training set only, and applies them to val and test —
preventing any leakage of val/test statistics into the training process.

Produces four complementary feature families:

    1. LEXICAL  — TF-IDF unigrams + bigrams (sparse, fast, strong baseline)
    2. METADATA — Numeric signals: review_length, word_count, star_rating,
                  avg_word_len, exclamation_count, question_count,
                  uppercase_ratio, has_title
    3. TEXTBLOB — Polarity and subjectivity scores (rule-based sentiment signal)
    4. ASPECTS  — Keyword-matched product aspect flags:
                  ui, performance, login, pricing, customer_support,
                  bugs, updates, content

Each family can be used independently or combined. The pipeline saves:
    outputs/features/
        tfidf_train.npz     sparse TF-IDF matrix (train)
        tfidf_val.npz       sparse TF-IDF matrix (val)
        tfidf_test.npz      sparse TF-IDF matrix (test)
        meta_train.csv      metadata features (train)
        meta_val.csv        metadata features (val)
        meta_test.csv       metadata features (test)
        textblob_train.csv  polarity + subjectivity (train)
        textblob_val.csv    polarity + subjectivity (val)
        textblob_test.csv   polarity + subjectivity (test)
        aspects_train.csv   aspect flags (train)
        aspects_val.csv     aspect flags (val)
        aspects_test.csv    aspect flags (test)
        labels_train.csv    sentiment labels aligned to feature rows
        labels_val.csv
        labels_test.csv
        tfidf_vectorizer.pkl  fitted TF-IDF vectorizer (for inference)
        feature_report.txt    human-readable validation report

WHY THESE FEATURES?
-------------------
• TF-IDF: strong sparse baseline; captures vocabulary differences between
  positive and negative reviews without needing embeddings or a GPU.
• Metadata: word_count and review_length correlate with sentiment intensity
  (1★ reviews tend to be longer and more detailed — confirmed in your EDA).
  These features are cheap and highly interpretable.
• TextBlob polarity/subjectivity: fast rule-based signal that can directly
  validate whether the star_rule labels are internally consistent.
• Aspects: extracting UI, performance, pricing mentions lets the downstream
  model learn sentiment *conditional* on product area — essential for the
  client's goal of "deeper, structured understanding of how users think".

LEAKAGE PREVENTION
------------------
• TF-IDF vectorizer is fit ONLY on train.csv. The same fitted object is
  applied to val and test — so IDF weights never see future documents.
• TextBlob and aspect features are document-level and stateless — no fitting
  required, so no leakage is possible.
• Metadata scaling (StandardScaler) is fit on train only.
• All fitted objects are saved as .pkl for use at inference time.

USAGE
-----
  # Default: reads splits/ directory produced by split_dataset.py
  python feature_engineering.py

  # Custom split directory
  python feature_engineering.py --splits-dir splits/v2

  # Custom output directory
  python feature_engineering.py --out-dir outputs/features/v2

  # TF-IDF tuning
  python feature_engineering.py --max-features 20000 --ngram-max 3

  # Skip a feature family
  python feature_engineering.py --no-textblob
  python feature_engineering.py --no-aspects

  # Dry-run: validate inputs and print feature stats without writing files
  python feature_engineering.py --dry-run

INSTALL
-------
  pip install pandas scikit-learn textblob scipy
  python -m textblob.download_corpora

===============================================================================
"""

import argparse
import logging
import pickle
import sys
import re
import warnings
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level    = logging.INFO,
    format   = "[%(asctime)s] %(levelname)s  %(message)s",
    datefmt  = "%Y-%m-%d %H:%M:%S",
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("feature_engineering.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# ASPECT KEYWORD DICTIONARY
# ─────────────────────────────────────────────
# Each key becomes a binary feature column: 1 if any keyword is found in the
# review text, 0 otherwise. Extend this dict freely for domain-specific needs.
ASPECT_KEYWORDS: dict[str, list[str]] = {
    "ui": [
        "ui", "interface", "design", "layout", "button", "screen",
        "theme", "dark mode", "font", "icon", "navigation", "menu",
        "visual", "look", "display", "ux",
    ],
    "performance": [
        "slow", "fast", "lag", "laggy", "crash", "freeze", "hang",
        "loading", "speed", "performance", "smooth", "responsive",
        "stuttering", "delay", "memory", "battery", "drain",
    ],
    "login": [
        "login", "log in", "sign in", "sign up", "account", "password",
        "authentication", "2fa", "two-factor", "otp", "sso",
        "session", "logout", "register", "credentials",
    ],
    "pricing": [
        "price", "pricing", "cost", "expensive", "cheap", "free",
        "subscription", "premium", "pay", "paid", "refund", "charge",
        "billing", "money", "dollar", "fee", "plan", "tier",
    ],
    "customer_support": [
        "support", "customer service", "help", "response", "reply",
        "contact", "ticket", "chat", "email", "team", "staff",
        "agent", "representative", "resolve", "complaint",
    ],
    "bugs": [
        "bug", "glitch", "error", "broken", "fix", "issue",
        "problem", "fault", "defect", "not working", "doesnt work",
        "stopped working", "failure", "corrupt", "crash",
    ],
    "updates": [
        "update", "version", "upgrade", "patch", "new feature",
        "changelog", "release", "latest", "update broke",
        "after update", "since update",
    ],
    "content": [
        "content", "answer", "response", "quality", "accurate",
        "wrong", "incorrect", "hallucinate", "made up", "outdated",
        "information", "knowledge", "training", "model",
    ],
}


# ─────────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(text: str) -> str:
    """
    Lightweight normalisation before TF-IDF vectorisation:
    - Lowercase
    - Strip URLs
    - Collapse whitespace
    We intentionally keep punctuation so the TF-IDF can pick up signals
    like "!!!" or "???" (captured separately in metadata features anyway).
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # remove URLs
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
# FEATURE FAMILY 1: TF-IDF
# ─────────────────────────────────────────────

def build_tfidf(
    train_texts: pd.Series,
    val_texts:   pd.Series,
    test_texts:  pd.Series,
    max_features: int = 15_000,
    ngram_max:    int = 2,
) -> tuple:
    """
    Fit TF-IDF on train only, transform all three splits.
    Returns (vectorizer, train_matrix, val_matrix, test_matrix).
    """
    log.info(f"  Fitting TF-IDF (max_features={max_features}, ngram=(1,{ngram_max}))...")

    vec = TfidfVectorizer(
        preprocessor  = preprocess,
        analyzer      = "word",
        ngram_range   = (1, ngram_max),
        max_features  = max_features,
        sublinear_tf  = True,       # log(1+tf) — reduces impact of very frequent terms
        min_df        = 3,          # ignore terms appearing in fewer than 3 docs
        max_df        = 0.95,       # ignore terms in >95% of docs (near stop-words)
        strip_accents = "unicode",
        token_pattern = r"(?u)\b\w\w+\b",  # at least 2 chars
    )

    X_train = vec.fit_transform(train_texts)
    X_val   = vec.transform(val_texts)
    X_test  = vec.transform(test_texts)

    log.info(f"  TF-IDF vocab size: {len(vec.vocabulary_):,}")
    log.info(f"  Train shape: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    log.info(f"  Train matrix density: {X_train.nnz / (X_train.shape[0]*X_train.shape[1]):.4%}")

    return vec, X_train, X_val, X_test


# ─────────────────────────────────────────────
# FEATURE FAMILY 2: METADATA
# ─────────────────────────────────────────────

def build_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract numeric metadata features from a review DataFrame.
    All features are document-level — no corpus-level fitting needed.
    """
    text = df["review_text"].fillna("").astype(str)

    meta = pd.DataFrame(index=df.index)

    # Core length signals (correlated with sentiment intensity per your EDA)
    meta["review_length"]    = df.get("review_length", text.str.len()).fillna(0).astype(int)
    meta["word_count"]       = df.get("word_count",    text.str.split().str.len()).fillna(0).astype(int)
    meta["avg_word_len"]     = text.apply(
        lambda t: np.mean([len(w) for w in t.split()]) if t.split() else 0.0
    )

    # Punctuation signals
    meta["exclamation_count"] = text.str.count(r"!")
    meta["question_count"]    = text.str.count(r"\?")
    meta["ellipsis_count"]    = text.str.count(r"\.\.\.")

    # Capitalisation signal (shouting = strong emotion)
    meta["uppercase_ratio"]  = text.apply(
        lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
    )

    # Star rating as a numeric feature (note: this is already your label proxy,
    # but useful as a feature for multi-task or regression heads)
    meta["star_rating"]      = pd.to_numeric(df.get("star_rating"), errors="coerce").fillna(0)

    # Apple-only: whether the review has a title (non-null title = more deliberate review)
    if "title" in df.columns:
        meta["has_title"]    = df["title"].notna().astype(int)
    else:
        meta["has_title"]    = 0

    return meta


def scale_metadata(
    train_meta: pd.DataFrame,
    val_meta:   pd.DataFrame,
    test_meta:  pd.DataFrame,
) -> tuple:
    """Fit StandardScaler on train, apply to all three."""
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_meta),
        columns = train_meta.columns,
    )
    val_scaled = pd.DataFrame(
        scaler.transform(val_meta),
        columns = val_meta.columns,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_meta),
        columns = test_meta.columns,
    )
    return scaler, train_scaled, val_scaled, test_scaled


# ─────────────────────────────────────────────
# FEATURE FAMILY 3: TEXTBLOB SENTIMENT
# ─────────────────────────────────────────────

def build_textblob(texts: pd.Series) -> pd.DataFrame:
    """
    Run TextBlob on each review and return polarity and subjectivity scores.

    polarity:     -1.0 (very negative) → +1.0 (very positive)
    subjectivity:  0.0 (very objective) → +1.0 (very subjective)

    These serve as lightweight rule-based features that are independent of
    the star_rule labels — useful for validating label quality and as
    complementary signals in the model.
    """
    try:
        from textblob import TextBlob
    except ImportError:
        log.warning("  TextBlob not installed. Run: pip install textblob && python -m textblob.download_corpora")
        empty = pd.DataFrame({"polarity": [0.0]*len(texts), "subjectivity": [0.0]*len(texts)})
        return empty

    polarity     = []
    subjectivity = []
    for text in texts:
        try:
            blob = TextBlob(str(text))
            polarity.append(blob.sentiment.polarity)
            subjectivity.append(blob.sentiment.subjectivity)
        except Exception:
            polarity.append(0.0)
            subjectivity.append(0.0)

    return pd.DataFrame({"polarity": polarity, "subjectivity": subjectivity})


# ─────────────────────────────────────────────
# FEATURE FAMILY 4: ASPECT FLAGS
# ─────────────────────────────────────────────

def build_aspects(texts: pd.Series) -> pd.DataFrame:
    """
    For each aspect in ASPECT_KEYWORDS, produce a binary column:
    1 if any keyword from that aspect appears in the review, else 0.

    Uses word-boundary regex so "UI" matches but "fluid" does not trigger
    the "ui" keyword.
    """
    # Pre-compile patterns for speed
    patterns = {
        aspect: re.compile(
            r"\b(" + "|".join(re.escape(kw) for kw in kws) + r")\b",
            flags=re.IGNORECASE,
        )
        for aspect, kws in ASPECT_KEYWORDS.items()
    }

    result = {}
    for aspect, pattern in patterns.items():
        result[f"aspect_{aspect}"] = texts.apply(
            lambda t: int(bool(pattern.search(str(t))))
        )

    return pd.DataFrame(result, index=texts.index)


# ─────────────────────────────────────────────
# VALIDATION REPORT
# ─────────────────────────────────────────────

def build_report(
    df_train, df_val, df_test,
    X_tfidf_train, X_tfidf_val, X_tfidf_test,
    meta_train, meta_val, meta_test,
    tb_train, tb_val, tb_test,
    asp_train, asp_val, asp_test,
    vec,
) -> str:
    lines = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines += [
        "=" * 65,
        " SCIENCIA AI — FEATURE ENGINEERING REPORT",
        f" Generated : {ts}",
        "=" * 65,
        "",
        "── SPLIT SIZES ──────────────────────────────────────────",
        f"  Train : {len(df_train):>8,} rows",
        f"  Val   : {len(df_val):>8,} rows",
        f"  Test  : {len(df_test):>8,} rows",
        "",
        "── SENTIMENT DISTRIBUTION ───────────────────────────────",
    ]
    for split_name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        dist = df["sentiment"].value_counts(normalize=True).mul(100).round(1).to_dict()
        lines.append(f"  {split_name:<6}: {dist}")

    lines += [
        "",
        "── TF-IDF ───────────────────────────────────────────────",
        f"  Vocabulary size     : {len(vec.vocabulary_):,}",
        f"  Train shape         : {X_tfidf_train.shape}",
        f"  Val   shape         : {X_tfidf_val.shape}",
        f"  Test  shape         : {X_tfidf_test.shape}",
        f"  Train matrix density: {X_tfidf_train.nnz / (X_tfidf_train.shape[0]*X_tfidf_train.shape[1]):.4%}",
        "",
        "── METADATA FEATURES ────────────────────────────────────",
        f"  Columns : {list(meta_train.columns)}",
        f"  Train mean review_length : {meta_train['review_length'].mean():.1f} chars",
        f"  Val   mean review_length : {meta_val['review_length'].mean():.1f} chars",
        f"  Test  mean review_length : {meta_test['review_length'].mean():.1f} chars",
        "",
        "── TEXTBLOB SENTIMENT ───────────────────────────────────",
        f"  Train polarity  : mean={tb_train['polarity'].mean():.3f}  std={tb_train['polarity'].std():.3f}",
        f"  Val   polarity  : mean={tb_val['polarity'].mean():.3f}  std={tb_val['polarity'].std():.3f}",
        f"  Test  polarity  : mean={tb_test['polarity'].mean():.3f}  std={tb_test['polarity'].std():.3f}",
        "",
        "── ASPECT FLAGS ─────────────────────────────────────────",
    ]

    for col in asp_train.columns:
        pct = asp_train[col].mean() * 100
        lines.append(f"  {col:<28}: {pct:.1f}% of train reviews mention this aspect")

    lines += [
        "",
        "── LABEL / FEATURE ALIGNMENT CHECK ─────────────────────",
    ]
    for split_name, df, meta, tb, asp in [
        ("Train", df_train, meta_train, tb_train, asp_train),
        ("Val",   df_val,   meta_val,   tb_val,   asp_val),
        ("Test",  df_test,  meta_test,  tb_test,  asp_test),
    ]:
        ok = (len(df) == len(meta) == len(tb) == len(asp))
        lines.append(f"  {split_name:<6}: {len(df):,} rows — all feature sets aligned: {'✓' if ok else '✗ MISMATCH'}")

    lines += ["", "=" * 65]
    return "\n".join(lines)


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(
    splits_dir:   Path,
    out_dir:      Path,
    max_features: int  = 15_000,
    ngram_max:    int  = 2,
    use_textblob: bool = True,
    use_aspects:  bool = True,
    dry_run:      bool = False,
) -> dict:

    log.info("=" * 65)
    log.info(" SCIENCIA AI — FEATURE ENGINEERING PIPELINE")
    log.info(f" Splits dir   : {splits_dir}")
    log.info(f" Output dir   : {out_dir}")
    log.info(f" Max TF-IDF   : {max_features:,}  ngram max: {ngram_max}")
    log.info(f" TextBlob     : {use_textblob}")
    log.info(f" Aspects      : {use_aspects}")
    log.info(f" Dry-run      : {dry_run}")
    log.info("=" * 65)

    # ── Load splits ───────────────────────────────────────────────────────────
    log.info("[1/6] Loading train / val / test splits...")
    for fname in ("train.csv", "val.csv", "test.csv"):
        p = splits_dir / fname
        if not p.exists():
            raise FileNotFoundError(
                f"Split file not found: {p}\n"
                f"Run split_dataset.py first to generate splits."
            )

    df_train = pd.read_csv(splits_dir / "train.csv")
    df_val   = pd.read_csv(splits_dir / "val.csv")
    df_test  = pd.read_csv(splits_dir / "test.csv")

    # Ensure review_text column exists (may be named 'review' in raw CSVs)
    for df in (df_train, df_val, df_test):
        if "review_text" not in df.columns and "review" in df.columns:
            df.rename(columns={"review": "review_text"}, inplace=True)

    log.info(f"  Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")

    # Validate sentiment column
    for split_name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        if "sentiment" not in df.columns:
            raise ValueError(f"'sentiment' column missing from {split_name}.csv")
        dist = df["sentiment"].value_counts().to_dict()
        log.info(f"  {split_name} sentiment: {dist}")

    if dry_run:
        log.info("[DRY-RUN] Input validation passed. No features computed or written.")
        return {"status": "dry_run"}

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── TF-IDF ────────────────────────────────────────────────────────────────
    log.info("[2/6] Building TF-IDF features...")
    vec, X_tfidf_train, X_tfidf_val, X_tfidf_test = build_tfidf(
        df_train["review_text"], df_val["review_text"], df_test["review_text"],
        max_features=max_features, ngram_max=ngram_max,
    )
    sparse.save_npz(out_dir / "tfidf_train.npz", X_tfidf_train)
    sparse.save_npz(out_dir / "tfidf_val.npz",   X_tfidf_val)
    sparse.save_npz(out_dir / "tfidf_test.npz",  X_tfidf_test)

    # Save the fitted vectorizer for inference
    with open(out_dir / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vec, f)
    log.info(f"  Saved tfidf_vectorizer.pkl → {out_dir}/")

    # ── Metadata ──────────────────────────────────────────────────────────────
    log.info("[3/6] Building metadata features...")
    meta_train_raw = build_metadata(df_train)
    meta_val_raw   = build_metadata(df_val)
    meta_test_raw  = build_metadata(df_test)

    scaler, meta_train, meta_val, meta_test = scale_metadata(
        meta_train_raw, meta_val_raw, meta_test_raw
    )
    with open(out_dir / "meta_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    meta_train.to_csv(out_dir / "meta_train.csv", index=False)
    meta_val.to_csv(out_dir   / "meta_val.csv",   index=False)
    meta_test.to_csv(out_dir  / "meta_test.csv",  index=False)
    log.info(f"  Metadata columns: {list(meta_train.columns)}")

    # ── TextBlob ──────────────────────────────────────────────────────────────
    if use_textblob:
        log.info("[4/6] Building TextBlob sentiment features (this may take a minute)...")
        tb_train = build_textblob(df_train["review_text"])
        tb_val   = build_textblob(df_val["review_text"])
        tb_test  = build_textblob(df_test["review_text"])
        tb_train.to_csv(out_dir / "textblob_train.csv", index=False)
        tb_val.to_csv(out_dir   / "textblob_val.csv",   index=False)
        tb_test.to_csv(out_dir  / "textblob_test.csv",  index=False)
        log.info(f"  Train polarity mean: {tb_train['polarity'].mean():.3f}")
    else:
        log.info("[4/6] TextBlob skipped (--no-textblob).")
        tb_train = pd.DataFrame({"polarity": [0.0]*len(df_train), "subjectivity": [0.0]*len(df_train)})
        tb_val   = pd.DataFrame({"polarity": [0.0]*len(df_val),   "subjectivity": [0.0]*len(df_val)})
        tb_test  = pd.DataFrame({"polarity": [0.0]*len(df_test),  "subjectivity": [0.0]*len(df_test)})

    # ── Aspects ───────────────────────────────────────────────────────────────
    if use_aspects:
        log.info("[5/6] Building aspect keyword features...")
        asp_train = build_aspects(df_train["review_text"])
        asp_val   = build_aspects(df_val["review_text"])
        asp_test  = build_aspects(df_test["review_text"])
        asp_train.to_csv(out_dir / "aspects_train.csv", index=False)
        asp_val.to_csv(out_dir   / "aspects_val.csv",   index=False)
        asp_test.to_csv(out_dir  / "aspects_test.csv",  index=False)
        for col in asp_train.columns:
            pct = asp_train[col].mean() * 100
            log.info(f"  {col:<28}: {pct:.1f}% coverage")
    else:
        log.info("[5/6] Aspects skipped (--no-aspects).")
        asp_train = pd.DataFrame(index=df_train.index)
        asp_val   = pd.DataFrame(index=df_val.index)
        asp_test  = pd.DataFrame(index=df_test.index)

    # ── Labels ────────────────────────────────────────────────────────────────
    log.info("[6/6] Writing aligned label files and validation report...")
    df_train[["review_id", "sentiment"]].to_csv(out_dir / "labels_train.csv", index=False)
    df_val[["review_id",   "sentiment"]].to_csv(out_dir / "labels_val.csv",   index=False)
    df_test[["review_id",  "sentiment"]].to_csv(out_dir / "labels_test.csv",  index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    report = build_report(
        df_train, df_val, df_test,
        X_tfidf_train, X_tfidf_val, X_tfidf_test,
        meta_train, meta_val, meta_test,
        tb_train, tb_val, tb_test,
        asp_train, asp_val, asp_test,
        vec,
    )
    report_path = out_dir / "feature_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print("\n" + report)

    log.info(f"\nAll features saved to: {out_dir}/")
    log.info("=" * 65)

    return {
        "status"       : "success",
        "tfidf_shape"  : X_tfidf_train.shape,
        "vocab_size"   : len(vec.vocabulary_),
        "meta_cols"    : list(meta_train.columns),
        "aspect_cols"  : list(asp_train.columns) if use_aspects else [],
        "n_train"      : len(df_train),
        "n_val"        : len(df_val),
        "n_test"       : len(df_test),
        "out_dir"      : str(out_dir),
    }


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description     = "Sciencia AI — Feature engineering pipeline",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog          = """
Examples:
  python feature_engineering.py
  python feature_engineering.py --splits-dir splits/v2 --out-dir outputs/features/v2
  python feature_engineering.py --max-features 20000 --ngram-max 3
  python feature_engineering.py --no-textblob
  python feature_engineering.py --no-aspects
  python feature_engineering.py --dry-run
        """,
    )
    parser.add_argument("--splits-dir",   default="splits",
                        help="Directory containing train/val/test CSVs (default: splits/)")
    parser.add_argument("--out-dir",      default="outputs/features",
                        help="Output directory for feature files (default: outputs/features/)")
    parser.add_argument("--max-features", type=int, default=15_000,
                        help="Max TF-IDF vocabulary size (default: 15000)")
    parser.add_argument("--ngram-max",    type=int, default=2,
                        help="Max n-gram size for TF-IDF (default: 2)")
    parser.add_argument("--no-textblob",  action="store_true",
                        help="Skip TextBlob sentiment features")
    parser.add_argument("--no-aspects",   action="store_true",
                        help="Skip aspect keyword features")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Validate inputs without computing or writing features")
    # parse_known_args silently ignores Jupyter's kernel arguments
    # (e.g. -f kernel-xxx.json) instead of crashing with SystemExit
    args, _unknown = parser.parse_known_args()

    run_pipeline(
        splits_dir   = Path(args.splits_dir),
        out_dir      = Path(args.out_dir),
        max_features = args.max_features,
        ngram_max    = args.ngram_max,
        use_textblob = not args.no_textblob,
        use_aspects  = not args.no_aspects,
        dry_run      = args.dry_run,
    )


# ─────────────────────────────────────────────
# JUPYTER-FRIENDLY DIRECT CALL
# ─────────────────────────────────────────────
# If you are running this inside a Jupyter notebook, call run_pipeline()
# directly instead of main() to avoid argument-parsing entirely. Example:
#
#   from feature_engineering import run_pipeline
#   from pathlib import Path
#
#   result = run_pipeline(
#       splits_dir   = Path("splits"),
#       out_dir      = Path("outputs/features"),
#       max_features = 15_000,
#       ngram_max    = 2,
#       use_textblob = True,
#       use_aspects  = True,
#       dry_run      = False,
#   )
#   print(result)

if __name__ == "__main__":
    main()
