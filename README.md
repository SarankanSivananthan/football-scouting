# Football Scouting Tool

Percentile radars and a player-similarity search built on real per-90 statistical profiles — the two things an actual scouting report is made of — plus an honest check on whether "playing-style archetypes" are statistically real or just a nice story.

**Stack**: Python, pandas, sklearn, Plotly, Jupyter

## 🎯 Goal

Two concrete tools scouts and analysts actually use:

1. **Percentile radar** — how does a player rank against others *in the same position*, on the metrics that define that position?
2. **Similarity search** — given a player, who has the closest statistical profile, purely from performance data (no reputation, no price)?

Plus a third, more skeptical question: is there real statistical structure behind labels like "poacher" or "creator," or is that just a convenient story laid on top of continuous data?

## 📊 Dataset

FBref's "Big 5 European Leagues" (Premier League, La Liga, Bundesliga, Serie A, Ligue 1) advanced season stats — standard, shooting, passing, possession, defense, goal/shot-creating actions (GCA), and miscellaneous — via [`worldfootballR_data`](https://github.com/JaseZiv/worldfootballR_data), the pre-scraped companion dataset to the R package `worldfootballR`. Using this instead of scraping FBref directly avoids hammering a site that sits behind Cloudflare specifically to discourage that.

**A data-quality check changed the project's season choice.** The most recent season in this dataset (2022-23) turned out to be an *incomplete* scrape — it tops out at 1,170 minutes played per player, about a third of a real season, instead of the ~3,400 a full-time starter reaches. Using it as the "current" snapshot would have quietly made every player look like a bench player. **The 2021-22 season is the most recent one with a complete scrape** (max minutes ≈3,420) and is what this project uses throughout — see `notebooks/analysis.ipynb` §1 for the check that caught this.

After filtering to outfield players (goalkeepers excluded — different metrics entirely) with **≥900 minutes** played (~10 full matches, the usual floor for per-90 comparisons to mean anything): **1,499 players** — 637 defenders, 597 midfielders, 265 forwards.

## 🏗 Methodology

### 1. From raw totals to comparable per-90 rates

FBref's columns are mostly season totals, useless for comparing a player with 900 minutes to one with 3,000. Every rate feature (`*_p90`) is computed by hand: `raw_count / (minutes_played / 90)`. 30 curated per-90 metrics span attacking output, passing/progression, dribbling/carrying, and defending — full list in [`src/similarity.py`](src/similarity.py).

Multi-club seasons (a player transferred mid-season) are handled by keeping only their largest stint by minutes — a documented simplification affecting ~2% of players, rather than weighted-averaging every rate across stints.

### 2. Percentile radars, position by position

A player's radar shows their **percentile rank within their own position group** on a curated, position-specific set of ~10 metrics ([`src/radar.py`](src/radar.py)) — a forward's radar (xG, box touches, dribbles) has nothing in common with a defender's (tackles, aerials, clearances), because comparing a centre-back's shot volume to a striker's would be meaningless. Metrics where *lower* is better (errors, fouls) have their percentile inverted, so every axis consistently reads "further out = better."

### 3. Similarity search

Standardize the full 30-feature profile **within each position group** (so a "high" tackle rate means something specific to defenders, not a global average across all positions), then rank candidates by cosine similarity via `sklearn.neighbors.NearestNeighbors` — the same idea as the [Spotify lyrics clustering project](https://github.com/SarankanSivananthan/spotify-lyrics-clustering)'s embedding similarity, applied to a hand-engineered stat profile instead of a learned embedding.

### 4. Do playing-style archetypes actually exist?

KMeans per position group, with a silhouette-score sweep over k=2..6 to check whether the clusters are statistically real before trusting the labels — same discipline as the K/eps selection in the Spotify project.

## 🔎 Key findings

- **Similarity search finds sensible neighbors with zero reputation signal in the input.** Virgil van Dijk's nearest neighbors (Aguerd, Acerbi, Fonte, Guéhi, Dunk) are all commanding, aerially dominant centre-backs; Kevin De Bruyne's (Insigne, Pléa, Hofmann, Mount, Berardi) are all creative attacking players who combine chance creation with carrying threat. Neither list was cherry-picked.
- **Forward playing styles don't separate into clean statistical clusters.** The silhouette sweep peaks at k=2 and stays low (≤0.22) throughout — real per-90 profiles are continuous, not naturally bucketed. Reported as a descriptive segmentation only, not a claim of statistically distinct "types," which mirrors the same honest finding in the Spotify project's DBSCAN section.
- **Position-specific radar templates aren't a cosmetic choice** — they're what makes a percentile mean anything. A single universal template would put a centre-back's shot volume on the same axis as a striker's, which is not a comparison anyone wants.

Full walkthrough with real outputs: [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb).

## 📂 Project structure

```
scripts/
  download_data.py    # fetches 7 FBref stat-category snapshots (~11MB, not committed)
src/
  load_data.py           # reads .rds files (pure-Python, no R install needed)
  build_features.py     # raw tables -> per-90 feature table (data/processed/players_2022.parquet)
  similarity.py           # position-grouped standardization + cosine-similarity search
  radar.py                  # percentile computation + position-specific radar charts
  style_clusters.py     # KMeans playing-style clustering with silhouette-score selection
notebooks/
  analysis.ipynb           # full walkthrough with real outputs
  build_notebook.py      # generates analysis.ipynb from source
```

## 🚀 Running it

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py   # ~11MB download, 7 stat-category files
python src/build_features.py         # builds data/processed/players_2022.parquet

jupyter notebook notebooks/analysis.ipynb
```

Or use it directly from Python:

```python
from src.similarity import PlayerSimilarity
from src.radar import compute_percentiles, player_radar, RADAR_TEMPLATES
import pandas as pd

df = pd.read_parquet("data/processed/players_2022.parquet")

sim = PlayerSimilarity(df)
sim.find_similar("Kevin De Bruyne", n=5)

df_pct = compute_percentiles(df, [c for t in RADAR_TEMPLATES.values() for c in t])
player_radar(df_pct, "Kylian Mbappé").show()
```
