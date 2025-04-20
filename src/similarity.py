"""
Finds players with a similar statistical profile to a given player, within
the same position group — same idea as the audio/lyrics similarity search
in the Spotify clustering project, applied to per-90 playing-style stats
instead of embeddings.

Each position group (DF / MF / FW) is standardized *separately*: a
defender's "high" tackle rate and a forward's "high" tackle rate are not
the same thing, so z-scoring within the position group is what makes
"most similar" mean something.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Every engineered per-90 rate / percentage feature is used for
# similarity — more signal than the curated radar subset, since this
# isn't meant to be read by a human at a glance the way a radar is.
FEATURE_COLS = [
    "goals_p90", "assists_p90", "npxg_p90", "xag_p90", "shots_p90", "sot_pct",
    "goals_minus_npxg_p90", "pass_cmp_pct", "prog_passes_p90", "key_passes_p90",
    "passes_into_box_p90", "touches_p90", "touches_att_pen_p90", "dribble_succ_pct",
    "dribbles_succ_p90", "prog_carries_p90", "carries_into_box_p90", "miscontrols_p90",
    "tackles_p90", "interceptions_p90", "tackles_plus_int_p90", "blocks_p90",
    "clearances_p90", "errors_p90", "sca_p90", "gca_p90", "fouls_p90", "fouled_p90",
    "recoveries_p90", "aerial_win_pct",
]


class PlayerSimilarity:
    def __init__(self, df: pd.DataFrame, feature_cols: list[str] = FEATURE_COLS):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self._models: dict[str, tuple[StandardScaler, NearestNeighbors, pd.DataFrame]] = {}
        for position, group in self.df.groupby("position"):
            scaler = StandardScaler().fit(group[feature_cols])
            X = scaler.transform(group[feature_cols])
            nn = NearestNeighbors(metric="cosine").fit(X)
            self._models[position] = (scaler, nn, group.reset_index(drop=True))

    def find_similar(self, player_name: str, n: int = 5) -> pd.DataFrame:
        rows = self.df[self.df["player"].str.lower() == player_name.lower()]
        if rows.empty:
            matches = self.df[self.df["player"].str.lower().str.contains(player_name.lower())]
            hint = f" Did you mean: {', '.join(matches['player'].head(5))}?" if len(matches) else ""
            raise ValueError(f"No player found matching '{player_name}'.{hint}")
        query = rows.iloc[0]
        position = query["position"]
        scaler, nn, group = self._models[position]

        query_vec = scaler.transform(query[self.feature_cols].to_frame().T)
        distances, indices = nn.kneighbors(query_vec, n_neighbors=n + 1)  # +1: query matches itself

        result = group.iloc[indices[0]].copy()
        result["similarity"] = 1 - distances[0]  # cosine distance -> similarity
        result = result[result["player"].str.lower() != player_name.lower()]
        return result[["player", "club", "league", "age", "position", "similarity"] + self.feature_cols].head(n)
