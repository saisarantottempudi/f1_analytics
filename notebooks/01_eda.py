"""EDA companion for data/processed/features.parquet.

Converted to a notebook via `jupytext` style — # %% cell markers. You can open
this as a notebook (VS Code / Jupyter with jupytext) or run it as a plain
script (`python notebooks/01_eda.py`) and it will drop charts into
notebooks/figures/.
"""

# %% [markdown]
# # F1 Analytics — Stage 3 EDA
#
# What we look at:
# 1. Dataset shape & missingness
# 2. Grid → finish relationship (per-circuit character)
# 3. Driver rolling form vs actual finish (does our form feature predict?)
# 4. Teammate head-to-head leaderboard
# 5. Circuit pole-to-win rate ranking

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path.cwd()
while not (ROOT / "data" / "processed").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
FEATURES = ROOT / "data" / "processed" / "features.parquet"
FIG_DIR = ROOT / "notebooks" / "figures"
FIG_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
df = pd.read_parquet(FEATURES)
print(f"{len(df):,} rows × {df.shape[1]} cols · seasons {df['season'].min()}–{df['season'].max()}")

# %% [markdown]
# ## 1. Missingness

# %%
missing = (df.isna().mean() * 100).sort_values(ascending=False)
missing = missing[missing > 0]
fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(missing))))
missing.plot.barh(ax=ax, color="#c44")
ax.set_xlabel("% null")
ax.set_title("Null rate per feature column")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(FIG_DIR / "01_missingness.png", dpi=120)
plt.close(fig)
print("saved 01_missingness.png")

# %% [markdown]
# ## 2. Grid vs finish — per-circuit character
# Tight correlation = grid-locked track (Monaco). Loose = chaotic (Interlagos wet, early Bahrain).

# %%
finishers = df[df["is_finish"] == 1].dropna(subset=["grid", "position"])
fig, ax = plt.subplots(figsize=(8, 7))
heat = pd.crosstab(finishers["grid"].clip(upper=20),
                   finishers["position"].clip(upper=20), normalize="index")
sns.heatmap(heat, cmap="magma", ax=ax, cbar_kws={"label": "P(finish | grid)"})
ax.set_xlabel("Finish position")
ax.set_ylabel("Grid position")
ax.set_title("Grid → finish transition probability")
fig.tight_layout()
fig.savefig(FIG_DIR / "02_grid_vs_finish.png", dpi=120)
plt.close(fig)
print("saved 02_grid_vs_finish.png")

# %% [markdown]
# ## 3. Rolling form vs actual — does the feature predict?

# %%
roll = df.dropna(subset=["drv_rollN_avg_finish_5", "position"])
fig, ax = plt.subplots(figsize=(9, 6))
sns.regplot(
    data=roll,
    x="drv_rollN_avg_finish_5", y="position",
    scatter_kws={"alpha": 0.25, "s": 20}, line_kws={"color": "#c44"}, ax=ax,
)
rho = roll[["drv_rollN_avg_finish_5", "position"]].corr().iloc[0, 1]
ax.set_title(f"Rolling-5 avg finish → actual finish   (ρ = {rho:.2f})")
ax.set_xlabel("Rolling 5-race avg finish (shifted)")
ax.set_ylabel("Actual finish position")
fig.tight_layout()
fig.savefig(FIG_DIR / "03_form_vs_actual.png", dpi=120)
plt.close(fig)
print(f"saved 03_form_vs_actual.png  ρ={rho:.2f}")

# %% [markdown]
# ## 4. Teammate head-to-head leaderboard (end-of-season, quali & race win rates)

# %%
last_round = df.groupby("season")["round"].transform("max")
end_of_season = df[df["round"] == last_round].copy()
h2h = (
    end_of_season.groupby(["driver_id", "driver_name"])
    [["tmate_quali_win_rate_season", "tmate_race_win_rate_season"]]
    .mean().dropna().sort_values("tmate_race_win_rate_season", ascending=False).head(15)
)
fig, ax = plt.subplots(figsize=(10, 7))
h2h.plot.barh(ax=ax)
ax.set_xlabel("Win rate vs teammate (season-to-date)")
ax.set_title("Top 15 drivers by teammate H2H")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(FIG_DIR / "04_teammate_h2h.png", dpi=120)
plt.close(fig)
print("saved 04_teammate_h2h.png")
print(h2h.to_string())

# %% [markdown]
# ## 5. Circuit priors: pole-to-win rate
# High bars = pole almost always converts (Monaco, Suzuka). Low = chaotic.

# %%
priors = (
    df[["circuit_id", "circ_pole_to_win_rate", "circ_grid_finish_corr"]]
    .drop_duplicates(subset="circuit_id")
    .dropna(subset=["circ_pole_to_win_rate"])
    .sort_values("circ_pole_to_win_rate", ascending=False)
    .head(20)
)
fig, ax = plt.subplots(figsize=(10, 8))
priors.plot.barh(x="circuit_id", y="circ_pole_to_win_rate", ax=ax, color="#345")
ax.set_xlabel("Pole-to-win conversion rate")
ax.set_title("Circuit priors — most 'pole-locked' tracks")
ax.invert_yaxis()
ax.get_legend().remove()
fig.tight_layout()
fig.savefig(FIG_DIR / "05_pole_to_win.png", dpi=120)
plt.close(fig)
print("saved 05_pole_to_win.png")
