"""
Downloads FBref "Big 5 European Leagues" advanced season stats, published
as pre-scraped .rds snapshots by the `worldfootballR_data` project
(https://github.com/JaseZiv/worldfootballR_data) — the companion dataset
to the popular R package `worldfootballR`. Using this instead of scraping
FBref directly avoids hammering their site (FBref sits behind Cloudflare
specifically to discourage that) and gets a dataset that's already clean
and structured.

Eight stat categories per player-season: standard, shooting, passing,
possession, defense, goal/shot-creating actions (GCA), miscellaneous, and
playing time. ~11MB total.

Run: python scripts/download_data.py
"""
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/JaseZiv/worldfootballR_data/master/data/fb_big5_advanced_season_stats"
FILES = [
    "big5_player_standard", "big5_player_shooting", "big5_player_passing",
    "big5_player_possession", "big5_player_defense", "big5_player_gca",
    "big5_player_misc", "big5_player_playing_time",
]

RAW_DIR = Path("data/raw")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        url = f"{BASE_URL}/{name}.rds"
        dest = RAW_DIR / f"{name}.rds"
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  -> {dest} ({dest.stat().st_size / 1e6:.1f}MB)")
    print("Done. Run `python src/build_features.py` next.")


if __name__ == "__main__":
    main()
