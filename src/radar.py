"""
Percentile radar charts — the standard scouting-report visual (the same
idea FBref's own "scouting report" pages and most public radar tools use):
for each metric, where does this player rank against others in the same
position, as a percentile from 0 to 100?

A curated, position-specific subset of metrics is used (not all 30
features from similarity.py) — a radar with 10 axes is readable, one with
30 is not.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# Metrics where a *lower* raw value is better (errors, fouls, miscontrols)
# get their percentile inverted so every axis on the radar reads
# "further out = better," consistently.
LOWER_IS_BETTER = {"errors_p90", "fouls_p90", "miscontrols_p90"}

RADAR_TEMPLATES = {
    "FW": {
        "npxg_p90": "Non-penalty xG",
        "goals_minus_npxg_p90": "Finishing (G - xG)",
        "shots_p90": "Shots",
        "sot_pct": "Shot accuracy %",
        "xag_p90": "xA",
        "touches_att_pen_p90": "Box touches",
        "dribbles_succ_p90": "Dribbles won",
        "prog_carries_p90": "Progressive carries",
        "aerial_win_pct": "Aerial win %",
        "sca_p90": "Shot-creating actions",
    },
    "MF": {
        "prog_passes_p90": "Progressive passes",
        "key_passes_p90": "Key passes",
        "passes_into_box_p90": "Passes into box",
        "pass_cmp_pct": "Pass completion %",
        "prog_carries_p90": "Progressive carries",
        "dribble_succ_pct": "Dribble success %",
        "tackles_plus_int_p90": "Tackles + interceptions",
        "sca_p90": "Shot-creating actions",
        "gca_p90": "Goal-creating actions",
        "recoveries_p90": "Recoveries",
    },
    "DF": {
        "tackles_plus_int_p90": "Tackles + interceptions",
        "interceptions_p90": "Interceptions",
        "clearances_p90": "Clearances",
        "blocks_p90": "Blocks",
        "aerial_win_pct": "Aerial win %",
        "pass_cmp_pct": "Pass completion %",
        "prog_passes_p90": "Progressive passes",
        "errors_p90": "Errors (inverted)",
        "recoveries_p90": "Recoveries",
        "fouls_p90": "Discipline (inverted fouls)",
    },
}


def compute_percentiles(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Percentile rank of every player within their own position group."""
    out = df.copy()
    for pos, group in df.groupby("position"):
        for col in feature_cols:
            pct = group[col].rank(pct=True) * 100
            if col in LOWER_IS_BETTER:
                pct = 100 - pct
            out.loc[group.index, f"{col}_pctile"] = pct
    return out


def player_radar(df_with_pctiles: pd.DataFrame, player_name: str) -> go.Figure:
    row = df_with_pctiles[df_with_pctiles["player"].str.lower() == player_name.lower()]
    if row.empty:
        raise ValueError(f"No player found matching '{player_name}'.")
    row = row.iloc[0]
    position = row["position"]
    template = RADAR_TEMPLATES[position]

    labels = list(template.values())
    values = [row[f"{col}_pctile"] for col in template]
    # Close the polygon.
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed, theta=labels_closed, fill="toself",
        name=row["player"], hovertemplate="%{theta}: %{r:.0f}th percentile<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=f"{row['player']} — {row['club']} ({row['league']}) — percentile vs. {position}s, 2021-22 season",
        showlegend=False,
    )
    return fig
