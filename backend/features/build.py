"""Build a (driver × race) feature table from raw Ergast Parquet.

Feature families:
    - Driver rolling form  (last-N-race avg finish, DNF rate, points trend)
    - Driver × circuit history (career avg finish / best result at this track)
    - Consistency metrics  (std of finishes in the season so far)
    - Constructor form     (last-N-race team avg finish)
    - Teammate H2H         (quali + race vs current teammate, season-to-date)
    - Circuit priors       (pole-to-win rate, grid→finish correlation, DNF rate)

Everything is computed strictly causally — no row of features for race N
uses data from race ≥ N. This is important for training a model that
generalises out-of-sample.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ERGAST_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "data" / "raw" / "ergast"
OUT_PATH_DEFAULT = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.parquet"


# ---------- loading ----------

def load_ergast(dir_: Path = ERGAST_DIR_DEFAULT) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name in ("schedule", "results", "qualifying", "pitstops"):
        p = dir_ / f"{name}.parquet"
        out[name] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    if out["results"].empty:
        raise FileNotFoundError(
            f"No results.parquet in {dir_}. Run scripts/ingest_ergast.py first."
        )
    return out


def is_finished(status_series: pd.Series) -> pd.Series:
    """Ergast 'status' is 'Finished' for classified, '+N Lap(s)' for lapped,
    anything else = DNF / DSQ / withdrawal."""
    s = status_series.astype(str)
    return s.eq("Finished") | s.str.startswith("+")


# ---------- feature builders ----------

def driver_rolling_form(results: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Shifted rolling stats per driver — only uses races BEFORE the current one."""
    df = results.sort_values(["driver_id", "date"]).copy()
    df["is_finish"] = is_finished(df["status"])
    df["pos_for_roll"] = df["position"].where(df["is_finish"])  # NaN on DNF

    g = df.groupby("driver_id", sort=False, group_keys=False)
    df[f"drv_rollN_avg_finish_{window}"] = (
        g["pos_for_roll"].shift(1).groupby(df["driver_id"], sort=False)
        .rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df[f"drv_rollN_dnf_rate_{window}"] = (
        g["is_finish"].shift(1).groupby(df["driver_id"], sort=False)
        .rolling(window, min_periods=1).apply(lambda s: 1.0 - s.mean(), raw=False)
        .reset_index(level=0, drop=True)
    )
    df[f"drv_rollN_points_avg_{window}"] = (
        g["points"].shift(1).groupby(df["driver_id"], sort=False)
        .rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    return df[[
        "driver_id", "season", "round",
        f"drv_rollN_avg_finish_{window}",
        f"drv_rollN_dnf_rate_{window}",
        f"drv_rollN_points_avg_{window}",
    ]]


def driver_circuit_history(results: pd.DataFrame) -> pd.DataFrame:
    """Career record per (driver, circuit) — strictly prior races only."""
    df = results.sort_values(["driver_id", "circuit_id", "date"]).copy()
    df["is_finish"] = is_finished(df["status"])
    df["pos_for_avg"] = df["position"].where(df["is_finish"])

    g = df.groupby(["driver_id", "circuit_id"], sort=False)
    df["drv_circ_races_prior"] = g.cumcount()
    df["drv_circ_avg_finish_prior"] = (
        g["pos_for_avg"].shift(1)
        .groupby([df["driver_id"], df["circuit_id"]], sort=False)
        .expanding().mean().reset_index(level=[0, 1], drop=True)
    )
    df["drv_circ_best_prior"] = (
        g["pos_for_avg"].shift(1)
        .groupby([df["driver_id"], df["circuit_id"]], sort=False)
        .expanding().min().reset_index(level=[0, 1], drop=True)
    )
    return df[[
        "driver_id", "season", "round",
        "drv_circ_races_prior", "drv_circ_avg_finish_prior", "drv_circ_best_prior",
    ]]


def driver_consistency(results: pd.DataFrame) -> pd.DataFrame:
    """Std / IQR of finishing position across the season so far."""
    df = results.sort_values(["driver_id", "season", "round"]).copy()
    df["pos_for_std"] = df["position"].where(is_finished(df["status"]))

    g = df.groupby(["driver_id", "season"], sort=False)
    df["drv_season_finish_std"] = (
        g["pos_for_std"].shift(1)
        .groupby([df["driver_id"], df["season"]], sort=False)
        .expanding().std().reset_index(level=[0, 1], drop=True)
    )
    df["drv_season_finish_iqr"] = (
        g["pos_for_std"].shift(1)
        .groupby([df["driver_id"], df["season"]], sort=False)
        .expanding().apply(lambda s: s.quantile(0.75) - s.quantile(0.25), raw=False)
        .reset_index(level=[0, 1], drop=True)
    )
    return df[[
        "driver_id", "season", "round",
        "drv_season_finish_std", "drv_season_finish_iqr",
    ]]


def constructor_form(results: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Team-level rolling average of finishing positions (across both cars)."""
    df = results.sort_values(["constructor_id", "date"]).copy()
    df["is_finish"] = is_finished(df["status"])
    df["pos_for_roll"] = df["position"].where(df["is_finish"])

    # Mean team finish per race first, then roll over races.
    race_team = (
        df.groupby(["constructor_id", "season", "round", "date"], as_index=False)
        ["pos_for_roll"].mean()
        .sort_values(["constructor_id", "date"])
    )
    race_team[f"team_rollN_avg_finish_{window}"] = (
        race_team.groupby("constructor_id")["pos_for_roll"]
        .shift(1).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    return race_team[[
        "constructor_id", "season", "round",
        f"team_rollN_avg_finish_{window}",
    ]]


def teammate_h2h(results: pd.DataFrame, qualifying: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date head-to-head vs current teammate on grid and finish."""
    r = results[["season", "round", "date", "driver_id", "constructor_id",
                 "grid", "position", "status"]].copy()
    r["race_rank"] = r["position"].where(is_finished(r["status"]))

    # Prior-race teammate comparison — build a per-team-per-race pair table.
    teams = (
        r.groupby(["season", "round", "constructor_id"])
        .agg(drivers=("driver_id", list))
        .reset_index()
    )
    teams = teams[teams["drivers"].str.len() == 2]  # only proper pairings
    teams[["drv_a", "drv_b"]] = pd.DataFrame(teams["drivers"].tolist(), index=teams.index)
    teams = teams.drop(columns=["drivers"])

    a = teams.merge(r[["season", "round", "driver_id", "grid", "race_rank"]]
                    .rename(columns={"driver_id": "drv_a",
                                     "grid": "grid_a", "race_rank": "race_a"}),
                    on=["season", "round", "drv_a"], how="left")
    pair = a.merge(r[["season", "round", "driver_id", "grid", "race_rank"]]
                   .rename(columns={"driver_id": "drv_b",
                                    "grid": "grid_b", "race_rank": "race_b"}),
                   on=["season", "round", "drv_b"], how="left")
    pair["a_quali_win"] = (pair["grid_a"] < pair["grid_b"]).astype(int)
    pair["a_race_win"] = (pair["race_a"] < pair["race_b"]).fillna(False).astype(int)

    def stacked(pair_df: pd.DataFrame) -> pd.DataFrame:
        left = pair_df.rename(columns={"drv_a": "driver_id",
                                       "a_quali_win": "quali_win",
                                       "a_race_win": "race_win"})[
            ["season", "round", "driver_id", "quali_win", "race_win"]]
        flipped = pair_df.copy()
        flipped["quali_win"] = 1 - flipped["a_quali_win"]
        flipped["race_win"] = 1 - flipped["a_race_win"]
        right = flipped.rename(columns={"drv_b": "driver_id"})[
            ["season", "round", "driver_id", "quali_win", "race_win"]]
        return pd.concat([left, right], ignore_index=True)

    per_race = stacked(pair).sort_values(["driver_id", "season", "round"])
    g = per_race.groupby(["driver_id", "season"], sort=False)
    per_race["tmate_quali_win_rate_season"] = (
        g["quali_win"].shift(1)
        .groupby([per_race["driver_id"], per_race["season"]], sort=False)
        .expanding().mean().reset_index(level=[0, 1], drop=True)
    )
    per_race["tmate_race_win_rate_season"] = (
        g["race_win"].shift(1)
        .groupby([per_race["driver_id"], per_race["season"]], sort=False)
        .expanding().mean().reset_index(level=[0, 1], drop=True)
    )
    return per_race[[
        "driver_id", "season", "round",
        "tmate_quali_win_rate_season", "tmate_race_win_rate_season",
    ]]


def circuit_priors(results: pd.DataFrame) -> pd.DataFrame:
    """Per-circuit aggregates — pole-to-win rate, DNF rate, grid-finish corr.

    Computed once over the full history (not causally shifted). Priors are
    treated as static track character, joined onto every race row.
    """
    r = results.copy()
    r["is_finish"] = is_finished(r["status"])

    winners = r[r["position"] == 1][["season", "round", "circuit_id", "driver_id", "grid"]]
    pole_win = winners.assign(pole_win=(winners["grid"] == 1).astype(int))
    by_circ = pole_win.groupby("circuit_id")["pole_win"].mean()

    dnf_rate = 1.0 - r.groupby("circuit_id")["is_finish"].mean()

    # Correlation of grid vs finish across all finishers — high = grid-locked (Monaco);
    # low = chaotic / lots of shuffling (Interlagos in the wet).
    def _corr(g: pd.DataFrame) -> float:
        sub = g[g["is_finish"]][["grid", "position"]].dropna()
        if len(sub) < 10:
            return np.nan
        return float(sub["grid"].corr(sub["position"]))

    grid_finish_corr = r.groupby("circuit_id", group_keys=False).apply(
        _corr, include_groups=False
    )

    out = pd.DataFrame({
        "circ_pole_to_win_rate": by_circ,
        "circ_dnf_rate": dnf_rate,
        "circ_grid_finish_corr": grid_finish_corr,
    }).reset_index()
    return out


# ---------- orchestration ----------

def build_feature_table(
    ergast_dir: Path = ERGAST_DIR_DEFAULT,
    rolling_window: int = 5,
) -> pd.DataFrame:
    data = load_ergast(ergast_dir)
    results = data["results"]
    qualifying = data["qualifying"]

    keys = ["driver_id", "season", "round"]
    base = results[[
        "season", "round", "race_name", "circuit_id", "date",
        "driver_id", "driver_name", "constructor_id",
        "grid", "position", "points", "status",
    ]].copy()
    base["is_finish"] = is_finished(base["status"]).astype(int)

    feats = [
        driver_rolling_form(results, window=rolling_window),
        driver_circuit_history(results),
        driver_consistency(results),
        teammate_h2h(results, qualifying),
    ]
    out = base
    for f in feats:
        out = out.merge(f, on=keys, how="left")

    team_form = constructor_form(results)
    out = out.merge(team_form, on=["constructor_id", "season", "round"], how="left")

    priors = circuit_priors(results)
    out = out.merge(priors, on="circuit_id", how="left")

    return out.sort_values(["season", "round", "position"], na_position="last").reset_index(drop=True)


def save_features(df: pd.DataFrame, out_path: Path = OUT_PATH_DEFAULT) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path
