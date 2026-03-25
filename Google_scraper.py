import time
import sqlite3
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime, timezone
from scipy import stats

warnings.filterwarnings('ignore')

# ── Plotting style ──────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.15)
plt.rcParams.update({
    'figure.dpi': 130,
    'axes.titleweight': 'bold',
    'axes.titlesize': 13,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

ACCENT   = '#4A90D9'   # primary blue
POS_COL  = '#27AE60'   # green  — positive
NEU_COL  = '#F39C12'   # amber  — neutral
NEG_COL  = '#E74C3C'   # red    — negative
SENT_PAL = {'positive': POS_COL, 'neutral': NEU_COL, 'negative': NEG_COL, 'unrated': '#95A5A6'}

print('Setup complete.')

from google_play_scraper import Sort, reviews as gplay_reviews

APP_ID       = 'com.openai.chatgpt'
APP_NAME     = 'ChatGPT'
TARGET       = 25_000   # overshoot so we comfortably clear 10k after cleaning (non-English drop is high)
BATCH_SIZE   = 200
DELAY_SEC    = 0.5

records          = []
seen_ids         = set()
continuation_tok = None
batch_num        = 0

print(f'Scraping {APP_NAME} ({APP_ID}) — target {TARGET:,} reviews...')
print('─' * 55)

while len(records) < TARGET:
    try:
        result, continuation_tok = gplay_reviews(
            APP_ID,
            lang               = 'en',
            country            = 'us',
            sort               = Sort.NEWEST,
            count              = BATCH_SIZE,
            continuation_token = continuation_tok,
        )
    except Exception as e:
        print(f'[Error] {e}')
        break

    if not result:
        print('No results — end of available reviews.')
        break

    new_ids = {r['reviewId'] for r in result} - seen_ids
    if not new_ids:
        print('No new review IDs in batch — all reviews collected.')
        break

    for r in result:
        rid = r.get('reviewId', '')
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        records.append({
            'review_id'  : rid,
            'app_name'   : APP_NAME,
            'user'       : r.get('userName', ''),
            'star_rating': r.get('score', None),
            'date'       : r['at'].strftime('%Y-%m-%d') if r.get('at') else '',
            'review'     : (r.get('content') or '').strip(),
        })

    batch_num += 1
    if batch_num % 10 == 0:
        print(f'  Batch {batch_num:>4} | +{len(new_ids):>3} new | Total: {len(records):>6,}')
    time.sleep(DELAY_SEC)

df_raw = pd.DataFrame(records)
df_raw.to_csv('chatgpt_reviews_raw.csv', index=False)
print(f'\nScrape complete — {len(df_raw):,} raw reviews saved to chatgpt_reviews_raw.csv')

from langdetect import detect, LangDetectException

MIN_CHARS = 3

def is_english(text):
    try:
        return detect(str(text)) == 'en'
    except LangDetectException:
        return True

audit = {}          # track rows dropped at each step
df    = df_raw.copy()

# ── Type coercions ───────────────────────────────────────────────────────────
df['star_rating'] = pd.to_numeric(df['star_rating'], errors='coerce')
df['date']        = pd.to_datetime(df['date'],        errors='coerce')

# Step-by-step drops
steps = [
    ('missing_date',       lambda d: d[d['date'].notna()]),
    ('null_empty_review',  lambda d: d[d['review'].notna() & (d['review'].str.strip() != '')]),
    ('too_short',          lambda d: d[d['review'].str.strip().str.len() >= MIN_CHARS]),
    ('non_english',        lambda d: d[d['review'].apply(is_english)]),
    ('duplicate_id',       lambda d: d.drop_duplicates(subset='review_id', keep='first')),
]

for step_name, fn in steps:
    before = len(df)
    df     = fn(df)
    dropped = before - len(df)
    audit[step_name] = dropped
    print(f'  [{step_name:<20}] dropped {dropped:>5,} rows  | remaining: {len(df):>6,}')

# Derived columns
df['review_length'] = df['review'].str.len()
df['word_count']    = df['review'].str.split().str.len()
df['sentiment']     = df['star_rating'].apply(
    lambda s: 'unrated'  if pd.isna(s)
    else     ('positive' if s >= 4 else ('negative' if s <= 2 else 'neutral'))
)
df = df.reset_index(drop=True)

df.to_csv('chatgpt_reviews_cleaned.csv', index=False)
print(f'\nCleaned dataset: {len(df):,} reviews  |  raw: {len(df_raw):,}  |  retention: {len(df)/len(df_raw)*100:.1f}%')