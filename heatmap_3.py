import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# ── 1. Load CSVs ──────────────────────────────────────────────────────────────
ANNUAL_PATH = os.path.join("output", "heatmap_normalized_values_annual.csv")
DEC_PATH = os.path.join("output", "heatmap_normalized_values_december.csv")
OUT_PATH = os.path.join("output", "heatmap_diff_dec_vs_annual.png")

PCT_COLS = ["Holiday %", "Business %", "Others %", "Total %", "Female %"]

df_annual = pd.read_csv(ANNUAL_PATH, index_col="Country / Region")
df_dec = pd.read_csv(DEC_PATH, index_col="Country / Region")

# ── 2. Inner join & sort by annual Total % descending ─────────────────────────
common = df_annual.index.intersection(df_dec.index)
df_annual = df_annual.loc[common, PCT_COLS]
df_dec = df_dec.loc[common, PCT_COLS]

# Sort rows by annual Total % descending (biggest annual senders at top)
df_annual = df_annual.sort_values("Total %", ascending=False)
df_dec = df_dec.loc[df_annual.index]

print(f"Loaded {len(common)} countries (inner join annual ∩ december)")

# ── 3. Compute difference matrix ──────────────────────────────────────────────
df_diff = df_dec - df_annual  # percentage point differences

# ── 4. Per-column symmetric normalization to [0, 1] ──────────────────────────
df_norm = pd.DataFrame(index=df_diff.index, columns=PCT_COLS, dtype=float)
for col in PCT_COLS:
    col_data = df_diff[col]
    abs_max = max(abs(col_data.min()), abs(col_data.max()))
    if abs_max == 0:
        df_norm[col] = 0.5
    else:
        col_norm = col_data / abs_max          # [-1, 1]
        df_norm[col] = (col_norm + 1) / 2     # [0, 1]

# ── 5. Custom diverging colormap: Red → Off-white → Green ────────────────────
colors = ["#d73027", "#f7f7f7", "#1a9850"]
cmap = mcolors.LinearSegmentedColormap.from_list("rdwgn", colors, N=256)

# ── 6. Build heatmap ──────────────────────────────────────────────────────────
n_countries = len(df_diff)
fig_height = max(12, n_countries * 0.22)
fig, ax = plt.subplots(figsize=(14, fig_height))

annot_data = df_diff.round(1).values

sns.heatmap(
    df_norm,
    annot=annot_data,
    fmt=".1f",
    cmap=cmap,
    vmin=0,
    vmax=1,
    linewidths=0.3,
    linecolor="white",
    annot_kws={"size": 7},
    ax=ax,
    cbar_kws={"label": "Normalized score (Red=Dec<Annual, Green=Dec>Annual)"},
    xticklabels=PCT_COLS,
)

ax.set_title(
    "Cambodia 2025 — December vs. Full-Year Visitor Profile Δ by Country\n"
    "(percentage point difference: December minus Annual; per-metric normalized)",
    fontsize=12,
    pad=12,
)
ax.set_xlabel("Metric", fontsize=9)
ax.set_ylabel("Country / Region", fontsize=9)
ax.tick_params(axis="x", labelsize=9)
ax.tick_params(axis="y", labelsize=7)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
plt.close()

print(f"Difference heatmap saved → {OUT_PATH}")
