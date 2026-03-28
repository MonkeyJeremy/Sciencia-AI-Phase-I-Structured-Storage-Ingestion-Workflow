"""
analyze_reviews.py
==================
Sciencia AI — Phase I: Database Analysis
Connects to reviews.db and produces a full descriptive + statistical analysis,
saved as a multi-panel PDF report.
"""

import sqlite3
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "axes.edgecolor": "#cccccc",
})

C_POS  = "#27AE60"
C_NEU  = "#E67E22"
C_NEG  = "#C0392B"
C_BLUE = "#2980B9"
C_GRAY = "#7F8C8D"
SENT_COLORS = {"positive": C_POS, "neutral": C_NEU, "negative": C_NEG}
STAR_COLORS = ["#C0392B", "#E74C3C", "#E67E22", "#2ECC71", "#27AE60"]

DB_PATH = Path("reviews.db")

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data from reviews.db ...")
conn = sqlite3.connect(DB_PATH)

df = pd.read_sql("""
    SELECT r.review_id, r.star_rating, r.review_date,
           r.review_text, r.review_length, r.word_count,
           sl.sentiment, u.username
    FROM reviews r
    JOIN sentiment_labels sl ON sl.review_fk = r.id
    JOIN users u              ON r.user_fk    = u.id
""", conn)
conn.close()

df["review_date"] = pd.to_datetime(df["review_date"])
df["month"]       = df["review_date"].dt.to_period("M").astype(str)

print(f"  Loaded {len(df):,} rows — {df['month'].nunique()} months "
      f"({df['review_date'].min().date()} → {df['review_date'].max().date()})")


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE 1  —  Overview & distributions
# ╚══════════════════════════════════════════════════════════════════════════════
print("\n[Page 1] Overview & distributions ...")

fig1, axes = plt.subplots(2, 3, figsize=(16, 9))
fig1.suptitle("ChatGPT · Google Play Reviews — Overview & Distributions",
              fontsize=14, fontweight="bold", y=1.01)

# 1a. Star rating bar
ax = axes[0, 0]
star_counts = df["star_rating"].value_counts().sort_index()
bars = ax.bar(star_counts.index.astype(str),
              star_counts.values, color=STAR_COLORS,
              edgecolor="white", linewidth=0.5, zorder=3)
for bar, val in zip(bars, star_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 80,
            f"{val:,}\n({val/len(df)*100:.1f}%)",
            ha="center", va="bottom", fontsize=8.5, color="#444")
ax.set_title("Star rating distribution")
ax.set_xlabel("Stars"); ax.set_ylabel("Reviews")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

# 1b. Sentiment donut
ax = axes[0, 1]
sent = df["sentiment"].value_counts()
wedge_colors = [SENT_COLORS.get(s, C_GRAY) for s in sent.index]
wedges, texts, autotexts = ax.pie(
    sent.values, labels=sent.index, colors=wedge_colors,
    autopct="%1.1f%%", startangle=90,
    wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
    pctdistance=0.78)
for t in autotexts:
    t.set_fontsize(9)
centre = plt.Circle((0, 0), 0.55, color="white")
ax.add_patch(centre)
ax.text(0, 0, f"{len(df):,}\nreviews", ha="center", va="center",
        fontsize=10, fontweight="bold", color="#333")
ax.set_title("Sentiment distribution")

# 1c. Word count tiers
ax = axes[0, 2]
bins   = [0, 4, 9, 19, df["word_count"].max()]
labels = ["< 5 words", "5–9 words", "10–19 words", "20+ words"]
colors = [C_NEG, C_NEU, C_BLUE, C_POS]
wc_counts = pd.cut(df["word_count"], bins=bins, labels=labels).value_counts()[labels]
bars = ax.bar(labels, wc_counts.values, color=colors,
              edgecolor="white", linewidth=0.5, zorder=3)
for bar, val in zip(bars, wc_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            f"{val/len(df)*100:.1f}%",
            ha="center", va="bottom", fontsize=9)
ax.set_title("Word count tiers")
ax.set_ylabel("Reviews")
ax.tick_params(axis="x", rotation=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

# 1d. Review length histogram
ax = axes[1, 0]
ax.hist(df["review_length"], bins=50, color=C_BLUE, alpha=0.85,
        edgecolor="white", linewidth=0.4, zorder=3)
ax.axvline(df["review_length"].median(), color=C_NEG, lw=1.8,
           linestyle="--", label=f"Median {df['review_length'].median():.0f} ch")
ax.axvline(df["review_length"].mean(),   color=C_NEU, lw=1.8,
           linestyle=":",  label=f"Mean {df['review_length'].mean():.0f} ch")
ax.set_title("Review length distribution (chars)")
ax.set_xlabel("Characters"); ax.set_ylabel("Reviews")
ax.legend(fontsize=8.5)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

# 1e. Length by sentiment box
ax = axes[1, 1]
order = ["positive", "neutral", "negative"]
palette = {s: SENT_COLORS[s] for s in order}
sns.boxplot(data=df, x="sentiment", y="review_length",
            order=order, palette=palette,
            flierprops={"marker": "o", "markersize": 2, "alpha": 0.3},
            linewidth=1.0, ax=ax)
for sent_val in order:
    med = df[df["sentiment"] == sent_val]["review_length"].median()
    ax.text(order.index(sent_val), med + 5, f"{med:.0f}",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax.set_title("Review length by sentiment")
ax.set_xlabel("Sentiment"); ax.set_ylabel("Characters")

# 1f. Descriptive stats table
ax = axes[1, 2]
ax.axis("off")
stats_data = []
for col in ["review_length", "word_count", "star_rating"]:
    s = df[col].describe()
    stats_data.append([
        col.replace("_", " ").title(),
        f"{s['mean']:.1f}", f"{s['50%']:.1f}",
        f"{s['std']:.1f}", f"{s['min']:.0f}", f"{s['max']:.0f}"
    ])
table = ax.table(
    cellText=stats_data,
    colLabels=["Metric", "Mean", "Median", "Std", "Min", "Max"],
    cellLoc="center", loc="center",
    bbox=[0, 0.25, 1, 0.7]
)
table.auto_set_font_size(False)
table.set_fontsize(9)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#dddddd")
    if r == 0:
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#F8F9FA")
ax.set_title("Descriptive statistics", pad=70)

fig1.tight_layout()
fig1.savefig("page1_overview.png", bbox_inches="tight", dpi=130)
plt.close(fig1)
print("  Saved page1_overview.png")


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE 2  —  Temporal & sentiment trends
# ╚══════════════════════════════════════════════════════════════════════════════
print("[Page 2] Temporal & sentiment trends ...")

monthly = (df.groupby(["month", "sentiment"])
             .size().unstack(fill_value=0)
             .reindex(columns=["positive", "neutral", "negative"], fill_value=0))

monthly_star = df.groupby("month")["star_rating"].agg(["mean", "count"])

fig2, axes = plt.subplots(2, 2, figsize=(16, 9))
fig2.suptitle("ChatGPT · Google Play Reviews — Temporal & Sentiment Trends",
              fontsize=14, fontweight="bold", y=1.01)

# 2a. Stacked bar — monthly volume by sentiment
ax = axes[0, 0]
x = np.arange(len(monthly))
ax.bar(x, monthly["positive"], label="Positive", color=C_POS, zorder=3)
ax.bar(x, monthly["neutral"],  label="Neutral",  color=C_NEU, bottom=monthly["positive"], zorder=3)
ax.bar(x, monthly["negative"], label="Negative", color=C_NEG,
       bottom=monthly["positive"] + monthly["neutral"], zorder=3)
ax.set_xticks(x); ax.set_xticklabels(monthly.index, rotation=20, ha="right")
ax.set_title("Monthly review volume by sentiment")
ax.set_ylabel("Reviews")
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

# 2b. Avg star rating per month
ax = axes[0, 1]
ax.bar(np.arange(len(monthly_star)), monthly_star["mean"],
       color=C_BLUE, zorder=3, edgecolor="white")
ax.set_xticks(np.arange(len(monthly_star)))
ax.set_xticklabels(monthly_star.index, rotation=20, ha="right")
ax.set_ylim(0, 5.5)
ax.axhline(4.0, color=C_POS, linestyle="--", lw=1.2, label="4.0★ threshold")
ax.axhline(2.5, color=C_NEG, linestyle="--", lw=1.2, label="2.5★ threshold")
for i, (avg, cnt) in enumerate(zip(monthly_star["mean"], monthly_star["count"])):
    ax.text(i, avg + 0.1, f"{avg:.2f}★\n(n={cnt:,})",
            ha="center", va="bottom", fontsize=8.5)
ax.set_title("Avg star rating by month")
ax.set_ylabel("Stars (1–5)")
ax.legend(fontsize=9)

# 2c. Positive % trend
ax = axes[1, 0]
pos_pct = monthly["positive"] / monthly.sum(axis=1) * 100
neg_pct = monthly["negative"] / monthly.sum(axis=1) * 100
ax.plot(monthly.index, pos_pct, "o-", color=C_POS, lw=2, markersize=7, label="% positive")
ax.plot(monthly.index, neg_pct, "s-", color=C_NEG, lw=2, markersize=7, label="% negative")
ax.fill_between(monthly.index, pos_pct, alpha=0.10, color=C_POS)
ax.fill_between(monthly.index, neg_pct, alpha=0.10, color=C_NEG)
ax.set_ylim(0, 105)
ax.set_title("Positive & negative % by month")
ax.set_ylabel("Percentage (%)")
ax.tick_params(axis="x", rotation=20)
ax.legend(fontsize=9)
for i, (p, n) in enumerate(zip(pos_pct, neg_pct)):
    ax.text(i, p + 2, f"{p:.1f}%", ha="center", fontsize=8, color=C_POS)
    ax.text(i, n + 2, f"{n:.1f}%", ha="center", fontsize=8, color=C_NEG)

# 2d. Daily review count (trend line)
ax = axes[1, 1]
daily = df.groupby("review_date").size().reset_index(name="count")
ax.bar(daily["review_date"], daily["count"], color=C_BLUE, alpha=0.6, width=1, label="Daily count")
if len(daily) > 3:
    z = np.polyfit(np.arange(len(daily)), daily["count"], 1)
    p = np.poly1d(z)
    ax.plot(daily["review_date"], p(np.arange(len(daily))),
            color=C_NEG, lw=2, linestyle="--", label="Trend")
ax.set_title("Daily review volume")
ax.set_ylabel("Reviews per day")
ax.tick_params(axis="x", rotation=20)
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

fig2.tight_layout()
fig2.savefig("page2_temporal.png", bbox_inches="tight", dpi=130)
plt.close(fig2)
print("  Saved page2_temporal.png")


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE 3  —  Statistical tests & correlations
# ╚══════════════════════════════════════════════════════════════════════════════
print("[Page 3] Statistical tests & correlations ...")

fig3, axes = plt.subplots(2, 3, figsize=(16, 9))
fig3.suptitle("ChatGPT · Google Play Reviews — Statistical Analysis",
              fontsize=14, fontweight="bold", y=1.01)

# 3a. Violin — review length by star rating
ax = axes[0, 0]
plot_df = df[df["star_rating"].notna()].copy()
plot_df["star_rating"] = plot_df["star_rating"].astype(int)
sns.violinplot(data=plot_df, x="star_rating", y="review_length",
               palette=STAR_COLORS, inner="quartile",
               linewidth=0.8, ax=ax)
ax.set_title("Review length by star rating (violin)")
ax.set_xlabel("Stars"); ax.set_ylabel("Characters")

# 3b. Correlation heatmap
ax = axes[0, 1]
num_cols = ["star_rating", "review_length", "word_count"]
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdYlGn",
            center=0, vmin=-1, vmax=1,
            linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
            annot_kws={"fontsize": 11})
ax.set_title("Correlation matrix")

# 3c. Kruskal-Wallis test result
ax = axes[0, 2]
ax.axis("off")
groups = [df[df["star_rating"] == s]["review_length"].values for s in [1, 2, 3, 4, 5]]
h_stat, p_val = stats.kruskal(*groups)
spearman_r, sp_p = stats.spearmanr(df["star_rating"], df["review_length"])
point_r, pr_p    = stats.pearsonr(df["review_length"], df["word_count"])

results = [
    ["Test", "Statistic", "p-value", "Conclusion"],
    ["Kruskal-Wallis\n(length ~ stars)", f"H = {h_stat:.2f}",
     f"p {'< 0.001' if p_val < 0.001 else f'= {p_val:.4f}'}",
     "Significant ✓" if p_val < 0.05 else "Not significant"],
    ["Spearman ρ\n(stars ~ length)", f"ρ = {spearman_r:.3f}",
     f"p {'< 0.001' if sp_p < 0.001 else f'= {sp_p:.4f}'}",
     "Sig. neg. corr ✓" if sp_p < 0.05 else "Not significant"],
    ["Pearson r\n(length ~ words)", f"r = {point_r:.3f}",
     f"p {'< 0.001' if pr_p < 0.001 else f'= {pr_p:.4f}'}",
     "Strong corr ✓" if pr_p < 0.05 else "Not significant"],
]

table = ax.table(cellText=results[1:], colLabels=results[0],
                 cellLoc="center", loc="center", bbox=[0, 0.1, 1, 0.8])
table.auto_set_font_size(False)
table.set_fontsize(8.5)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#dddddd")
    if r == 0:
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#F8F9FA")
ax.set_title("Statistical tests", pad=50)

# 3d. Scatter — review_length vs word_count (sampled)
ax = axes[1, 0]
sample = df.sample(min(3000, len(df)), random_state=42)
colors_map = [SENT_COLORS[s] for s in sample["sentiment"]]
ax.scatter(sample["word_count"], sample["review_length"],
           c=colors_map, alpha=0.25, s=8, zorder=3)
m, b = np.polyfit(df["word_count"], df["review_length"], 1)
xr = np.linspace(df["word_count"].min(), df["word_count"].max(), 100)
ax.plot(xr, m * xr + b, color="#2C3E50", lw=1.8, linestyle="--",
        label=f"y = {m:.1f}x + {b:.0f}")
ax.set_title("Word count vs review length (n=3k sample)")
ax.set_xlabel("Word count"); ax.set_ylabel("Characters")
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color=C_POS, label="Positive"),
    Patch(color=C_NEU, label="Neutral"),
    Patch(color=C_NEG, label="Negative"),
    plt.Line2D([0], [0], color="#2C3E50", linestyle="--", label="Trend line"),
], fontsize=8, loc="upper left")

# 3e. Median length per star (bar)
ax = axes[1, 1]
med_per_star = df.groupby("star_rating")["review_length"].median()
bars = ax.bar(med_per_star.index.astype(str), med_per_star.values,
              color=STAR_COLORS, edgecolor="white", zorder=3)
for bar, val in zip(bars, med_per_star.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.0f} ch", ha="center", va="bottom", fontsize=9)
ax.set_title("Median review length by star rating")
ax.set_xlabel("Stars"); ax.set_ylabel("Median characters")

# 3f. Review length CDF by sentiment
ax = axes[1, 2]
for sent_val, color in SENT_COLORS.items():
    data = np.sort(df[df["sentiment"] == sent_val]["review_length"].values)
    cdf  = np.arange(1, len(data) + 1) / len(data)
    ax.plot(data, cdf, color=color, lw=2,
            label=f"{sent_val.title()} (n={len(data):,})")
ax.axvline(50, color=C_GRAY, linestyle=":", lw=1.2, label="50 ch mark")
ax.set_title("CDF — review length by sentiment")
ax.set_xlabel("Characters"); ax.set_ylabel("Cumulative proportion")
ax.set_xlim(0, 520); ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)

fig3.tight_layout()
fig3.savefig("page3_stats.png", bbox_inches="tight", dpi=130)
plt.close(fig3)
print("  Saved page3_stats.png")


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE 4  —  Data quality & pipeline audit
# ╚══════════════════════════════════════════════════════════════════════════════
print("[Page 4] Data quality audit ...")

fig4, axes = plt.subplots(2, 3, figsize=(16, 9))
fig4.suptitle("ChatGPT · Google Play Reviews — Data Quality Audit",
              fontsize=14, fontweight="bold", y=1.01)

# 4a. User frequency distribution
ax = axes[0, 0]
user_freq = df["username"].value_counts()
freq_groups = pd.cut(user_freq, bins=[0, 1, 2, 5, user_freq.max()],
                     labels=["1 review", "2 reviews", "3–5 reviews", "6+ reviews"])
fg_counts = freq_groups.value_counts().sort_index()
ax.bar(fg_counts.index.astype(str), fg_counts.values,
       color=[C_BLUE, C_NEU, C_NEG, "#8E44AD"], edgecolor="white", zorder=3)
for i, (label, val) in enumerate(fg_counts.items()):
    ax.text(i, val + 10, f"{val:,}", ha="center", va="bottom", fontsize=9)
ax.set_title("User review frequency")
ax.set_ylabel("Number of users")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

# 4b. Review length category breakdown
ax = axes[0, 1]
bins2   = [0, 9, 19, 49, 99, 199, 500]
labels2 = ["0–9", "10–19", "20–49", "50–99", "100–199", "200–500"]
len_cat = pd.cut(df["review_length"], bins=bins2, labels=labels2).value_counts()[labels2]
ax.barh(labels2, len_cat.values, color=C_BLUE, alpha=0.85, edgecolor="white", zorder=3)
for i, val in enumerate(len_cat.values):
    ax.text(val + 20, i, f"{val:,}  ({val/len(df)*100:.1f}%)",
            va="center", fontsize=9)
ax.set_title("Review length buckets")
ax.set_xlabel("Reviews")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

# 4c. Null / anomaly count
ax = axes[0, 2]
ax.axis("off")
checks = [
    ("Total reviews",           len(df),                          ""),
    ("Unique users",            df["username"].nunique(),          ""),
    ("Null star_rating",        df["star_rating"].isna().sum(),   "✓ None" if df["star_rating"].isna().sum()==0 else "⚠ Present"),
    ("Emoji-only reviews",      int((df["review_length"] <= 5).sum()), "⚠ Low signal" if (df["review_length"]<=5).sum() > 0 else ""),
    ("< 5 words",               int((df["word_count"] < 5).sum()),    f"{(df['word_count']<5).mean()*100:.1f}%"),
    ("500-char truncations",    int((df["review_length"] == 500).sum()), "Hard limit hit"),
    ("Multi-review users",      int((df["username"].value_counts() > 1).sum()), ""),
    ("Date range (days)",       (df["review_date"].max() - df["review_date"].min()).days, ""),
]
cell_text = [[str(k), f"{v:,}" if isinstance(v, int) else str(v), n] for k, v, n in checks]
t = ax.table(cellText=cell_text, colLabels=["Check", "Value", "Note"],
             cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
t.auto_set_font_size(False); t.set_fontsize(9)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor("#ddd")
    if r == 0:
        cell.set_facecolor("#2C3E50"); cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#F8F9FA")
ax.set_title("Quality checklist", pad=5)

# 4d. Star × sentiment heatmap
ax = axes[1, 0]
cross = df.groupby(["star_rating", "sentiment"]).size().unstack(fill_value=0)
cross = cross.reindex(columns=["positive", "neutral", "negative"], fill_value=0)
sns.heatmap(cross, annot=True, fmt=",", cmap="YlGn",
            linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
            annot_kws={"fontsize": 9})
ax.set_title("Star rating × sentiment (count)")
ax.set_xlabel("Sentiment"); ax.set_ylabel("Stars")

# 4e. Review length percentiles
ax = axes[1, 1]
percentiles = np.arange(0, 101, 5)
pct_vals    = np.percentile(df["review_length"], percentiles)
ax.plot(percentiles, pct_vals, color=C_BLUE, lw=2, marker="o", markersize=4)
ax.fill_between(percentiles, pct_vals, alpha=0.15, color=C_BLUE)
ax.axhline(50, color=C_NEG, linestyle="--", lw=1.2, label="50 ch")
ax.axhline(100, color=C_NEU, linestyle="--", lw=1.2, label="100 ch")
for pct, val in [(25, np.percentile(df["review_length"], 25)),
                 (50, np.percentile(df["review_length"], 50)),
                 (75, np.percentile(df["review_length"], 75)),
                 (90, np.percentile(df["review_length"], 90))]:
    ax.annotate(f"P{pct}={val:.0f}", xy=(pct, val),
                xytext=(pct + 2, val + 15), fontsize=8.5,
                arrowprops={"arrowstyle": "->", "lw": 0.8})
ax.set_title("Review length percentile curve")
ax.set_xlabel("Percentile"); ax.set_ylabel("Characters")
ax.legend(fontsize=9)

# 4f. Top 10 power users
ax = axes[1, 2]
top_users = df["username"].value_counts().head(10)
colors_u = [C_NEG if v >= 6 else C_NEU if v >= 3 else C_BLUE for v in top_users.values]
ax.barh(top_users.index[::-1], top_users.values[::-1],
        color=colors_u[::-1], edgecolor="white", zorder=3)
for i, val in enumerate(top_users.values[::-1]):
    ax.text(val + 0.1, i, str(val), va="center", fontsize=9)
ax.set_title("Top 10 most active users")
ax.set_xlabel("Reviews posted")
ax.set_xlim(0, top_users.max() + 3)

fig4.tight_layout()
fig4.savefig("page4_quality.png", bbox_inches="tight", dpi=130)
plt.close(fig4)
print("  Saved page4_quality.png")


# ── Print summary stats to stdout ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ANALYSIS SUMMARY")
print("=" * 60)
print(f"  Reviews          : {len(df):,}")
print(f"  Date range       : {df['review_date'].min().date()} → {df['review_date'].max().date()}")
print(f"  Avg star rating  : {df['star_rating'].mean():.2f}")
print(f"  Avg length       : {df['review_length'].mean():.1f} ch")
print(f"  Avg word count   : {df['word_count'].mean():.1f} words")
print(f"\n  Sentiment split:")
for s, n in df["sentiment"].value_counts().items():
    print(f"    {s:<12}: {n:>6,}  ({n/len(df)*100:.1f}%)")
p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
print(f"\n  Kruskal-Wallis H = {h_stat:.2f}, p {p_str}")
print(f"  Spearman ρ       = {spearman_r:.3f} (stars ~ length)")
print(f"  Pearson  r       = {point_r:.3f} (length ~ word count)")
print("\n  All 4 chart pages saved.")
print("=" * 60)
