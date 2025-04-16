"""Builds analysis.ipynb from scratch as a sequence of markdown/code cells."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

md = lambda src: cells.append(nbf.v4.new_markdown_cell(src))
code = lambda src: cells.append(nbf.v4.new_code_cell(src))

md("""\
# Football Scouting Tool — Percentile Radars & Player Similarity

**Goal**: turn raw per-90 stats into two things scouts actually use —
**percentile radars** ("how does this player rank against their position
peers, on the metrics that matter for that position?") and a
**similarity search** ("who plays like this player?") — plus a look at
whether statistically distinct *playing-style archetypes* exist within a
position at all.

**Dataset**: FBref "Big 5 European Leagues" advanced season stats
(standard, shooting, passing, possession, defense, goal/shot-creating
actions, miscellaneous), via the [`worldfootballR_data`](https://github.com/JaseZiv/worldfootballR_data)
project. Full feature-engineering pipeline in `src/build_features.py` —
this notebook loads its output.
""")

code("""\
import sys
sys.path.insert(0, "../src")

import pandas as pd
import plotly.express as px

from similarity import PlayerSimilarity, FEATURE_COLS
from radar import compute_percentiles, player_radar, RADAR_TEMPLATES
from style_clusters import sweep_k, cluster_position_group
from sklearn.preprocessing import StandardScaler

df = pd.read_parquet("../data/processed/players_2022.parquet")
print(f"{len(df):,} players — {df['position'].value_counts().to_dict()}")
""")

md("""\
## 1. A data-quality check worth being upfront about

This dataset's most recent season (2022-23) is actually an **incomplete
scrape** — it tops out at 1,170 minutes played per player, about a third
of a full season, instead of the ~3,400 a ever-present starter reaches.
Using it as-is would have quietly made every player in the "current"
season look like a squad-rotation player. The **2021-22 season is the
most recent one with a complete scrape** (max minutes ≈3,420, matching a
real full season), so that's what this project uses throughout.
""")

code("""\
from load_data import load_table
standard = load_table("big5_player_standard")
print(standard.groupby("Season_End_Year")["Min_Playing"].max())
""")

md("""\
## 2. Feature engineering: raw counts to comparable per-90 rates

Most FBref columns are season totals, which aren't comparable across
players with different playing time. Every rate feature here (`*_p90`) is
computed by hand as `raw_count / (minutes_played / 90)`, restricted to
players with **≥900 minutes** in the season (~10 full matches — the usual
floor for including a player in per-90 comparisons at all, otherwise a
player's one hot game skews their rate wildly). Goalkeepers are excluded
entirely — their metrics live in a different world.

Full curated feature list: `src/similarity.py::FEATURE_COLS` (30 per-90
rates and percentages across attacking, passing, carrying, and defending).
""")

code("""\
df[["player","club","league","position","age","minutes"] + FEATURE_COLS[:6]].sample(5, random_state=3)
""")

md("""\
## 3. Percentile radars

For each metric, where does a player rank against others **in the same
position**, as a percentile? A curated, position-specific subset of ~10
metrics keeps each radar readable (see `src/radar.py::RADAR_TEMPLATES`).
""")

code("""\
all_template_cols = sorted({col for t in RADAR_TEMPLATES.values() for col in t})
df_pct = compute_percentiles(df, all_template_cols)
""")

code("""\
player_radar(df_pct, "Kylian Mbappé").show()
""")

code("""\
player_radar(df_pct, "Virgil van Dijk").show()
""")

md("""\
Mbappé's radar is what you'd expect for a 2021-22-season superstar
forward: elite on non-penalty xG, box touches and progressive carries.
Van Dijk's is a defender's radar, not a forward's — high on
tackles+interceptions, aerials and pass completion, and that's the point:
the *same ten-metric template* would be meaningless applied across
positions, which is why the radar is position-specific rather than
one-size-fits-all.
""")


nb["cells"] = cells
nbf.write(nb, "analysis.ipynb")
print("Wrote analysis.ipynb with", len(cells), "cells")
