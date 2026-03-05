"""
download_big5.py
----------------
Downloads Big 5 league CSVs from football-data.co.uk
Leagues : Premier League, Bundesliga, La Liga, Serie A, Ligue 1
Seasons : 2015/16 → 2025/26  (55 files total)

Usage:
    pip install requests tqdm
    python download_big5.py

Files are saved to ./data/{league}/{league}_{season}.csv
e.g.  data/E0/E0_1516.csv   ← Premier League 2015/16
      data/D1/D1_2526.csv   ← Bundesliga 2025/26
"""

import os
import time
import requests
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL   = "https://www.football-data.co.uk/mmz4281"
OUTPUT_DIR = Path("data")
DELAY_SEC  = 1.2    # polite delay between requests — don't hammer the server

LEAGUES = {
    "E0":  "England - Premier League",
    "D1":  "Germany - Bundesliga",
    "SP1": "Spain   - La Liga",
    "I1":  "Italy   - Serie A",
    "F1":  "France  - Ligue 1",
}

SEASONS = [
    "1516",  # 2015/16
    "1617",  # 2016/17
    "1718",  # 2017/18
    "1819",  # 2018/19
    "1920",  # 2019/20
    "2021",  # 2020/21
    "2122",  # 2021/22
    "2223",  # 2022/23
    "2324",  # 2023/24
    "2425",  # 2024/25
    "2526",  # 2025/26  ← current, partial season
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; football-data-downloader/1.0)"
}

# ── Download ──────────────────────────────────────────────────────────────────

def download(url: str, dest: Path) -> str:
    """Returns 'downloaded', 'skipped', 'missing', or 'error'."""
    if dest.exists():
        return "skipped"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 404 or "html" in r.headers.get("Content-Type", ""):
            return "missing"
        r.raise_for_status()
        if len(r.content) < 100:
            return "missing"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return "downloaded"
    except requests.RequestException:
        return "error"


def main():
    tasks = [
        (league, season)
        for season in SEASONS
        for league in LEAGUES
    ]

    counts = {"downloaded": 0, "skipped": 0, "missing": 0, "error": 0}

    print(f"\n⚽  Downloading Big 5 leagues — {len(tasks)} files\n")

    with tqdm(tasks, unit="file") as bar:
        for league, season in bar:
            url  = f"{BASE_URL}/{season}/{league}.csv"
            dest = OUTPUT_DIR / league / f"{league}_{season}.csv"
            bar.set_description(f"{LEAGUES[league]} {season[:2]}/{season[2:]}")

            result = download(url, dest)
            counts[result] += 1

            if result == "downloaded":
                tqdm.write(f"  ✅  {dest}")
            elif result == "error":
                tqdm.write(f"  ❌  Failed: {url}")
            elif result == "missing":
                tqdm.write(f"  ⚠️   Not found: {url}")

            if result != "skipped":
                time.sleep(DELAY_SEC)

    print(f"""
{'─' * 45}
  ✅  Downloaded : {counts['downloaded']}
  ⏭   Skipped   : {counts['skipped']}  (already existed)
  ⚠️   Missing   : {counts['missing']}
  ❌  Errors    : {counts['error']}
{'─' * 45}

Files saved to: {OUTPUT_DIR.resolve()}/
Structure:
  data/
    E0/   ← Premier League (11 files)
    D1/   ← Bundesliga     (11 files)
    SP1/  ← La Liga        (11 files)
    I1/   ← Serie A        (11 files)
    F1/   ← Ligue 1        (11 files)
""")


if __name__ == "__main__":
    main()