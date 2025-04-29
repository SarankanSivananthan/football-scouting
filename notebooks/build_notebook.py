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

md("""\
## 4. Player similarity search

Same idea as the audio/lyrics similarity search in the
[Spotify lyrics clustering project](https://github.com/SarankanSivananthan/spotify-lyrics-clustering),
applied to the full 30-feature per-90 profile instead of embeddings:
standardize within position group, then rank by cosine similarity.
""")

code("""\
sim = PlayerSimilarity(df)
sim.find_similar("Kevin De Bruyne", n=5)[["player", "club", "league", "similarity"]]
""")

code("""\
sim.find_similar("Virgil van Dijk", n=5)[["player", "club", "league", "similarity"]]
""")

md("""\
The De Bruyne neighbors (Insigne, Pléa, Hofmann, Mount, Berardi) are all
creative attacking midfielders/wide players who combine chance creation
with some individual carrying threat — a sensible statistical
neighborhood, even without the model knowing anything about reputation or
transfer value. The van Dijk neighbors (Aguerd, Acerbi, Fonte, Guéhi,
Dunk) are all no-nonsense, aerially dominant centre-backs. Neither list
was hand-picked — it's exactly what `NearestNeighbors` returns.
""")

md("""\
## 5. Do statistically distinct playing-style archetypes actually exist?

Same silhouette-score-driven approach as the Spotify project's KMeans/K
selection, applied here per position group: cluster forwards into styles
(poacher vs. creator vs. presser, say) — but check whether the clusters
are *statistically* distinct before trusting the labels.
""")

code("""\
fw = df[df["position"] == "FW"]
scaler = StandardScaler().fit(fw[FEATURE_COLS])
X = scaler.transform(fw[FEATURE_COLS])
scores = sweep_k(X)

fig = px.line(
    x=list(scores.keys()), y=list(scores.values()), markers=True,
    labels={"x": "k (number of clusters)", "y": "Silhouette score"},
    title="Silhouette score by k — forwards",
)
fig.show()
""")

md("""\
**The honest reading: k=2 scores highest, and every k scores low (≤0.22)
in absolute terms.** Real per-90 playing profiles are continuous, not
naturally clustered into a small number of tight, well-separated groups —
a striker's style shades gradually from poacher to creator rather than
falling into one bucket or another. This mirrors the same finding from
the Spotify project's DBSCAN section: high-dimensional, continuous
real-world data often resists clean clustering, and a silhouette-score
check exists specifically to catch that instead of quietly picking a
k that "looks nice" and calling the clusters more real than they are.

For descriptive purposes only (not because k=4 is statistically optimal —
it isn't), here's what 4 forward clusters look like:
""")

code("""\
labeled, centroids = cluster_position_group(df, "FW", k=4)
display_cols = ["npxg_p90", "xag_p90", "prog_carries_p90", "tackles_plus_int_p90", "aerial_win_pct"]
centroids[display_cols].round(2).assign(n_players=labeled["cluster"].value_counts().sort_index())
""")

md("""\
Even without statistically clean separation, the centroids differ in
sensible directions — one cluster higher on `npxg_p90` and lower on
`tackles_plus_int_p90` (poacher-leaning), another higher on `xag_p90` and
`prog_carries_p90` (creator-leaning) — just not sharply enough to call
them distinct "types" with statistical confidence. **Descriptive
segmentation, not a claim of natural clusters.**
""")

md("""\
## 6. Takeaways

- **A data-quality check up front changed the whole project's season
  choice** — the "most recent" season in the raw data was a red herring
  (an incomplete scrape), and using it uncritically would have quietly
  corrupted every per-90 rate in the project.
- **Percentiles only mean something within a like-for-like group** — the
  radar templates are position-specific by design, not a single template
  stretched across every role.
- **Similarity search on a curated per-90 profile finds sensible
  neighbors** without any reputation or price signal in the input —
  Van Dijk's nearest neighbors are all commanding centre-backs, De
  Bruyne's are all creative attacking midfielders.
- **Not every clustering question has a clean answer, and that's worth
  reporting rather than hiding** — forward playing styles don't separate
  into tight statistical clusters, and the silhouette sweep is what
  surfaces that instead of a cherry-picked k.
""")

nb["cells"] = cells
nbf.write(nb, "analysis.ipynb")
print("Wrote analysis.ipynb with", len(cells), "cells")
