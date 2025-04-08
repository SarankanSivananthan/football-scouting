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


nb["cells"] = cells
nbf.write(nb, "analysis.ipynb")
print("Wrote analysis.ipynb with", len(cells), "cells")
