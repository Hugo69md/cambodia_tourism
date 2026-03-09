import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# ── 1. Paths ──────────────────────────────────────────────────────────────────
ANNUAL_PATH = os.path.join("output", "06_arrivals_by_country_annual.csv")
DEC_PATH    = os.path.join("output", "10_arrivals_by_country_december.csv")
OUT_PATH    = os.path.join("output", "heatmap_diff_absolute_dec_vs_expected.png")

METRICS = ["Holiday", "Business", "Others", "Total", "Female"]
METRIC_COLS = [f"2025* {m}" for m in METRICS]

# Substrings that identify regional aggregates (case-insensitive)
AGGREGATE_KEYWORDS = [
    "grand total", "asia and", "asean", "northeast asia", "southern asia",
    "oceania", "europe", "northern europe", "western europe", "central / eastern",
    "east mediterranean", "other europe", "americas", "north america",
    "central america", "south america", "other america", "africa",
    "north africa", "subsaharan", "other africa", "middle east",
    "other middle east", "other asia",
]


def is_aggregate(name: str) -> bool:
    n = name.lower()
    return any(kw in n for kw in AGGREGATE_KEYWORDS)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[~df["Country / Region"].apply(is_aggregate)].copy()
    for col in METRIC_COLS:
        df[col] = (
            df[col].astype(str).str.replace(",", "", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.set_index("Country / Region")[METRIC_COLS]
    df.columns = METRICS
    return df


# ── 2. December seasonal weight (hard-coded, verified from CSV 02) ────────────
dec_total_2025    = 396_911
annual_total_2025 = 5_569_752
december_weight   = dec_total_2025 / annual_total_2025

print(f"December seasonal weight: {december_weight:.4f} ({december_weight * 100:.2f}%)")

# ── 3. Load data ──────────────────────────────────────────────────────────────
df_annual = load_csv(ANNUAL_PATH)
df_dec    = load_csv(DEC_PATH)

print(f"Loaded {len(df_annual)} countries from annual CSV (after filtering aggregates)")
print(f"Loaded {len(df_dec)} countries from December CSV (after filtering aggregates)")

# ── 4. Compute expected December from annual data ─────────────────────────────
df_expected = df_annual * december_weight

# ── 5. Inner join ─────────────────────────────────────────────────────────────
common = df_annual.index.intersection(df_dec.index)
df_expected = df_expected.loc[common]
df_dec_common = df_dec.loc[common]

print(f"Inner join: {len(common)} countries")

# ── 6. Sort by annual Total descending ────────────────────────────────────────
sort_order = df_annual.loc[common, "Total"].sort_values(ascending=False).index
df_expected    = df_expected.loc[sort_order]
df_dec_common  = df_dec_common.loc[sort_order]

# ── 7. Compute difference ─────────────────────────────────────────────────────
df_diff = df_dec_common - df_expected  # positive → more than expected (GREEN)

# ── 8. Per-column symmetric normalization to [0, 1] ──────────────────────────
df_norm = pd.DataFrame(index=df_diff.index, columns=METRICS, dtype=float)
for col in METRICS:
    col_data = df_diff[col]
    abs_max = max(abs(col_data.min()), abs(col_data.max()))
    if abs_max == 0:
        df_norm[col] = 0.5
    else:
        col_norm = col_data / abs_max       # [-1, 1]
        df_norm[col] = (col_norm + 1) / 2  # [0, 1]

# ── 9. Annotation matrix (rounded to integers) ───────────────────────────────
annot_matrix = df_diff.round(0).astype(int).astype(str).values

# ── 10. Custom diverging colormap: Red → Off-white → Green ───────────────────
colors = ["#d73027", "#f7f7f7", "#1a9850"]
cmap = LinearSegmentedColormap.from_list("rdwgn", colors, N=256)

# ── 11. Build heatmap ─────────────────────────────────────────────────────────
n_countries = len(df_diff)
fig, ax = plt.subplots(figsize=(14, max(12, n_countries * 0.22)))

sns.heatmap(
    df_norm,
    annot=annot_matrix,
    fmt="",
    cmap=cmap,
    vmin=0, vmax=1,
    linewidths=0.3,
    linecolor="lightgrey",
    ax=ax,
    cbar_kws={"label": "Normalized score (Red=Below expected, Green=Above expected)"},
    annot_kws={"size": 7},
)

ax.set_title(
    f"Cambodia 2025 — December Actual vs. Seasonally-Expected Visitor Counts \u0394 by Country\n"
    f"(diff = actual Dec 2025 \u2212 expected Dec 2025; expected = annual \u00d7 Dec/Total weight {december_weight:.2%})",
    fontsize=12,
    pad=12,
)
ax.set_xlabel("Metric", fontsize=9)
ax.set_ylabel("Country / Region", fontsize=9)
ax.set_xticklabels(METRICS)
ax.tick_params(axis="x", labelsize=9)
ax.tick_params(axis="y", labelsize=7)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
plt.close()

print(f"Heatmap saved \u2192 {OUT_PATH}")
