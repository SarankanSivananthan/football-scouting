"""
Builds the scouting feature table: one row per player for the 2021-22
season — the most recent season in this dataset with a *complete* scrape
(2022-23 is present but cuts off partway through, topping out at 1,170
minutes played per player instead of a full season's ~3,400 — using it
would silently penalize every player for a season that was never finished
scraping) — with every counting stat converted to a true per-90 rate.

Why per-90 and not raw totals: a midfielder with 900 minutes and one with
2,700 minutes can't be compared on raw tackle counts — only on tackles
*per 90 minutes played*. Most of FBref's own per-90 columns only exist for
a handful of headline stats, so most of the rates here are computed by
hand: raw_count / (minutes_played / 90).

Run: python src/build_features.py
Output: data/processed/players_2022.parquet
"""
from __future__ import annotations

import pandas as pd

from load_data import load_all

SEASON = 2022  # last season in this dataset with a complete scrape (2023 cuts off mid-season)
MIN_MINUTES = 900  # ~10 full matches — the usual scouting-analytics floor
                     # for including a player in per-90 comparisons at all
OUT = "data/processed/players_2022.parquet"


def dedupe_multi_club_stints(df: pd.DataFrame, minutes_col: str) -> pd.DataFrame:
    """A player transferred mid-season gets one row per club. Rather than
    weighted-averaging every rate column across stints (a lot of
    complexity for ~2% of players), we keep only the stint where they
    played the most minutes — a simple, clearly-documented simplification."""
    return (
        df.sort_values(minutes_col, ascending=False)
        .drop_duplicates(subset="Url", keep="first")
        .reset_index(drop=True)
    )


def per90(count: pd.Series, minutes: pd.Series) -> pd.Series:
    return (count / (minutes / 90)).astype(float)


def build_players_table() -> pd.DataFrame:
    tables = load_all()

    standard = tables["big5_player_standard"]
    standard = standard[standard["Season_End_Year"] == SEASON].copy()
    standard = dedupe_multi_club_stints(standard, "Min_Playing")
    standard = standard[standard["Min_Playing"] >= MIN_MINUTES].copy()
    standard["Age"] = pd.to_numeric(standard["Age"], errors="coerce")

    keep_urls = set(standard["Url"])

    def prep(name: str, cols: list[str]) -> pd.DataFrame:
        df = tables[name]
        df = df[df["Season_End_Year"] == SEASON]
        df = df[df["Url"].isin(keep_urls)]
        # These tables carry Mins_Per_90 (minutes/90) rather than raw
        # minutes — still monotonic with playing time, so it's the right
        # key to pick each player's largest stint by.
        df = dedupe_multi_club_stints(df, "Mins_Per_90")
        return df[["Url"] + cols].set_index("Url")

    shooting = prep("big5_player_shooting", [
        "Sh_Standard", "SoT_Standard", "SoT_percent_Standard",
        "G_per_Sh_Standard", "npxG_per_Sh_Expected", "Dist_Standard",
    ])
    passing = prep("big5_player_passing", [
        "Cmp_percent_Total", "PrgDist_Total", "KP", "Final_Third", "PPA", "CrsPA", "Prog",
    ])
    possession = prep("big5_player_possession", [
        "Touches_Touches", "Att Pen_Touches", "Succ_Dribbles", "Att_Dribbles",
        "Succ_percent_Dribbles", "Carries_Carries", "PrgDist_Carries", "Prog_Carries",
        "CPA_Carries", "Mis_Carries", "Dis_Carries",
    ])
    defense = prep("big5_player_defense", [
        "Tkl_Tackles", "TklW_Tackles", "Int", "Tkl+Int", "Blocks_Blocks", "Clr", "Err",
    ])
    gca = prep("big5_player_gca", ["SCA90_SCA", "GCA90_GCA"])
    misc = prep("big5_player_misc", ["Fls", "Fld", "Recov", "Won_Aerial", "Lost_Aerial", "Won_percent_Aerial"])

    df = standard.set_index("Url")
    for extra in [shooting, passing, possession, defense, gca, misc]:
        df = df.join(extra, how="left")
    df = df.reset_index()

    m90 = df["Min_Playing"] / 90

    # --- per-90 rates (feature engineering: raw counts -> comparable rates) ---
    df["goals_p90"] = per90(df["Gls"], df["Min_Playing"])
    df["assists_p90"] = per90(df["Ast"], df["Min_Playing"])
    df["npxg_p90"] = per90(df["npxG_Expected"], df["Min_Playing"])
    df["xag_p90"] = per90(df["xAG_Expected"], df["Min_Playing"])
    df["shots_p90"] = per90(df["Sh_Standard"], df["Min_Playing"])
    df["sot_pct"] = df["SoT_percent_Standard"]
    df["goals_minus_npxg_p90"] = df["goals_p90"] - df["npxg_p90"]  # finishing over/under-performance

    df["pass_cmp_pct"] = df["Cmp_percent_Total"]
    df["prog_passes_p90"] = per90(df["Prog"], df["Min_Playing"])
    df["key_passes_p90"] = per90(df["KP"], df["Min_Playing"])
    df["passes_into_box_p90"] = per90(df["PPA"], df["Min_Playing"])

    df["touches_p90"] = per90(df["Touches_Touches"], df["Min_Playing"])
    df["touches_att_pen_p90"] = per90(df["Att Pen_Touches"], df["Min_Playing"])
    df["dribble_succ_pct"] = df["Succ_percent_Dribbles"]
    df["dribbles_succ_p90"] = per90(df["Succ_Dribbles"], df["Min_Playing"])
    df["prog_carries_p90"] = per90(df["Prog_Carries"], df["Min_Playing"])
    df["carries_into_box_p90"] = per90(df["CPA_Carries"], df["Min_Playing"])
    df["miscontrols_p90"] = per90(df["Mis_Carries"] + df["Dis_Carries"], df["Min_Playing"])

    df["tackles_p90"] = per90(df["Tkl_Tackles"], df["Min_Playing"])
    df["interceptions_p90"] = per90(df["Int"], df["Min_Playing"])
    df["tackles_plus_int_p90"] = per90(df["Tkl+Int"], df["Min_Playing"])
    df["blocks_p90"] = per90(df["Blocks_Blocks"], df["Min_Playing"])
    df["clearances_p90"] = per90(df["Clr"], df["Min_Playing"])
    df["errors_p90"] = per90(df["Err"], df["Min_Playing"])

    df["sca_p90"] = df["SCA90_SCA"]
    df["gca_p90"] = df["GCA90_GCA"]

    df["fouls_p90"] = per90(df["Fls"], df["Min_Playing"])
    df["fouled_p90"] = per90(df["Fld"], df["Min_Playing"])
    df["recoveries_p90"] = per90(df["Recov"], df["Min_Playing"])
    df["aerial_win_pct"] = df["Won_percent_Aerial"]

    # Primary position (drop "GK" entirely; goalkeeper metrics are a
    # different world and out of scope for an outfield scouting tool).
    df["position"] = df["Pos"].str.split(",").str[0]
    df = df[df["position"].isin(["DF", "MF", "FW"])].copy()

    df = df.fillna({
        col: 0.0 for col in df.columns if df[col].dtype.kind in "fc"
    })

    identity_cols = ["Player", "Url", "Squad", "Comp", "Nation", "position", "Age", "Min_Playing", "MP_Playing"]
    feature_cols = [c for c in df.columns if c.endswith(("_p90", "_pct")) or c in (
        "goals_minus_npxg_p90",
    )]

    final = df[identity_cols + feature_cols].rename(columns={
        "Player": "player", "Url": "url", "Squad": "club", "Comp": "league",
        "Nation": "nation", "Age": "age", "Min_Playing": "minutes", "MP_Playing": "matches",
    })
    return final.reset_index(drop=True)


if __name__ == "__main__":
    import os

    table = build_players_table()
    os.makedirs("data/processed", exist_ok=True)
    table.to_parquet(OUT, index=False)
    print(f"Wrote {len(table):,} players, {table.shape[1]} columns to {OUT}")
    print(table["position"].value_counts())
