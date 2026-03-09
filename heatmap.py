import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ── 1. Load & clean ──────────────────────────────────────────────────────────
CSV_PATH = os.path.join("output", "06_arrivals_by_country_annual.csv")
OUT_PATH = os.path.join("output", "heatmap_visitor_profile_2025.png")

df = pd.read_csv(CSV_PATH)

NUMERIC_COLS = [
    "2024 Total",
    "2025* Holiday",
    "2025* Business",
    "2025* Others",
    "2025* Total",
    "2025* Female",
]

for col in NUMERIC_COLS:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace("", np.nan)
        .astype(float)
    )

# Drop rows where 2025* Total is 0 or NaN
df = df[df["2025* Total"].notna() & (df["2025* Total"] != 0)].copy()

# ── 2. Row filtering ──────────────────────────────────────────────────────────
AGGREGATES = {
    "Grand Total",
    "Asia and the Pacific",
    "ASEAN (Southeast Asia)",
    "Northeast Asia",
    "Southern Asia",
    "Oceania",
    "Other Asia & the Pacific",
    "Europe",
    "Northern Europe",
    "Western Europe",
    "Central / Eastern Europe",
    "Southern Europe",
    "East Mediterranean Europe",
    "Other Europe",
    "Americas",
    "North America",
    "Central America",
    "South America",
    "Other America",
    "Africa",
    "North Africa",
    "Subsaharan Africa",
    "Other Africa",
    "Middle East",
    "Other Middle East",
}

# Read Grand Total before filtering
grand_total_row = df[df["Country / Region"].str.strip() == "Grand Total"]
if grand_total_row.empty:
    raise ValueError("'Grand Total' row not found in CSV — cannot normalize Total %.")
grand_total = grand_total_row["2025* Total"].values[0]

# Drop aggregate rows and empty country names
df = df[
    df["Country / Region"].notna()
    & (~df["Country / Region"].str.strip().isin(AGGREGATES))
].copy()

print(f"Loaded {len(df)} country rows (after filtering aggregates)")

# ── 3. Normalization ──────────────────────────────────────────────────────────
total = df["2025* Total"]

df["Holiday %"] = df["2025* Holiday"] / total * 100
df["Business %"] = df["2025* Business"] / total * 100
df["Others %"] = df["2025* Others"] / total * 100
df["Total %"] = total / grand_total * 100
df["Female %"] = df["2025* Female"] / total * 100

PCT_COLS = ["Holiday %", "Business %", "Others %", "Total %", "Female %"]

# Sort by Total % descending
df = df.sort_values("Total %", ascending=False).reset_index(drop=True)

df_pct = df.set_index("Country / Region")[PCT_COLS]

# ── 4. Column-wise normalization to [0, 1] for uniform colormap ──────────────
df_norm = df_pct.apply(
    lambda c: (c - c.min()) / (c.max() - c.min()) if c.max() != c.min() else pd.Series(0.5, index=c.index)
)

# ── 5. Build heatmap ──────────────────────────────────────────────────────────
n_countries = len(df_pct)
fig_height = max(12, n_countries * 0.22)
fig, ax = plt.subplots(figsize=(14, fig_height))

sns.heatmap(
    df_norm,
    annot=df_pct.values,
    fmt=".1f",
    cmap="YlOrRd",
    vmin=0,
    vmax=1,
    linewidths=0.3,
    linecolor="white",
    annot_kws={"size": 7},
    ax=ax,
    cbar_kws={"label": "Relative intensity (column-normalised)"},
)

ax.set_title(
    "Cambodia 2025 — Visitor Profile by Country of Residence\n(% normalized per metric)",
    fontsize=12,
    pad=12,
)
ax.set_xlabel("Visitor Type", fontsize=9)
ax.set_ylabel("Country / Region", fontsize=9)
ax.tick_params(axis="x", labelsize=9)
ax.tick_params(axis="y", labelsize=7)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
plt.close()

print(f"Heatmap saved → {OUT_PATH}")
