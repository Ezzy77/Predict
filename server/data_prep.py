"""
data_prep.py — tailored for football-data.co.uk Premier League CSV (E0 format)
Columns confirmed present: Div, Date, Time, HomeTeam, AwayTeam, FTHG, FTAG, FTR,
  HTHG, HTAG, HTR, Referee, HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR,
  B365H, B365D, B365A, AvgH, AvgD, AvgA, Avg>2.5, Avg<2.5, ... (132 cols total)
"""

import os
from pathlib import Path

import pandas as pd
import numpy as np


# ── Team name normalization (variants → canonical) ─────────────────────────────
# Extend as needed when team names differ across seasons
TEAM_ALIASES: dict[str, str] = {}


# ── Column mapping: raw CSV → internal names ──────────────────────────────────
COLUMN_MAP = {
    # Teams & result
    'HomeTeam': 'home',
    'AwayTeam': 'away',
    'FTR':      'result',       # H / D / A

    # Full-time goals
    'FTHG': 'home_goals',
    'FTAG': 'away_goals',

    # Half-time goals
    'HTHG': 'ht_home_goals',
    'HTAG': 'ht_away_goals',
    'HTR':  'ht_result',

    # Match stats
    'HS':  'home_shots',
    'AS':  'away_shots',
    'HST': 'home_sot',
    'AST': 'away_sot',
    'HC':  'home_corners',
    'AC':  'away_corners',
    'HF':  'home_fouls',
    'AF':  'away_fouls',
    'HY':  'home_yellows',
    'AY':  'away_yellows',
    'HR':  'home_reds',
    'AR':  'away_reds',

    # Bet365 odds
    'B365H': 'odds_h',
    'B365D': 'odds_d',
    'B365A': 'odds_a',

    # Market average odds
    'AvgH': 'avg_odds_h',
    'AvgD': 'avg_odds_d',
    'AvgA': 'avg_odds_a',

    # Over/under market
    'Avg>2.5': 'avg_odds_over25',
    'Avg<2.5': 'avg_odds_under25',
    'B365>2.5': 'b365_over25',
    'B365<2.5': 'b365_under25',
}


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preparation steps to a raw DataFrame (already renamed).
    Expects columns: Date, home, away, result, home_goals, away_goals, etc.
    """
    # Parse date (DD/MM/YYYY or DD/MM/YY)
    parsed_date = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')
    df = pd.concat([df, parsed_date.rename('date')], axis=1)
    df = df.dropna(subset=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Cast numerics
    numeric_cols = [
        'home_goals', 'away_goals',
        'ht_home_goals', 'ht_away_goals',
        'home_shots', 'away_shots',
        'home_sot', 'away_sot',
        'home_corners', 'away_corners',
        'home_fouls', 'away_fouls',
        'home_yellows', 'away_yellows',
        'home_reds', 'away_reds',
        'odds_h', 'odds_d', 'odds_a',
        'avg_odds_h', 'avg_odds_d', 'avg_odds_a',
        'avg_odds_over25', 'avg_odds_under25',
        'b365_over25', 'b365_under25',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Derived columns
    derived = {}
    derived['total_goals'] = df['home_goals'] + df['away_goals']
    derived['over_2_5'] = (derived['total_goals'] > 2.5).astype(int)
    derived['over_1_5'] = (derived['total_goals'] > 1.5).astype(int)
    derived['over_3_5'] = (derived['total_goals'] > 3.5).astype(int)
    derived['btts'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)

    if {'ht_home_goals', 'ht_away_goals'}.issubset(df.columns):
        derived['ht_total_goals'] = df['ht_home_goals'] + df['ht_away_goals']
        derived['ht_over_0_5'] = (derived['ht_total_goals'] > 0.5).astype(int)

    derived['result_encoded'] = df['result'].map({'H': 2, 'D': 1, 'A': 0})

    if {'odds_h', 'odds_d', 'odds_a'}.issubset(df.columns):
        overround = (1/df['odds_h']) + (1/df['odds_d']) + (1/df['odds_a'])
        derived['implied_h'] = (1 / df['odds_h']) / overround
        derived['implied_d'] = (1 / df['odds_d']) / overround
        derived['implied_a'] = (1 / df['odds_a']) / overround

    if {'avg_odds_h', 'avg_odds_d', 'avg_odds_a'}.issubset(df.columns):
        avg_overround = (1/df['avg_odds_h']) + (1/df['avg_odds_d']) + (1/df['avg_odds_a'])
        derived['avg_implied_h'] = (1 / df['avg_odds_h']) / avg_overround
        derived['avg_implied_d'] = (1 / df['avg_odds_d']) / avg_overround
        derived['avg_implied_a'] = (1 / df['avg_odds_a']) / avg_overround

    if {'avg_odds_over25', 'avg_odds_under25'}.issubset(df.columns):
        ou_overround = (1/df['avg_odds_over25']) + (1/df['avg_odds_under25'])
        derived['implied_over25'] = (1 / df['avg_odds_over25']) / ou_overround

    df = pd.concat([df, pd.DataFrame(derived, index=df.index)], axis=1)

    # Drop rows missing essentials
    essential = ['home', 'away', 'date', 'home_goals', 'away_goals', 'result']
    df = df.dropna(subset=essential).reset_index(drop=True)

    # Ensure league exists for Big 5 (legacy pl_*.csv won't have it)
    if "league" not in df.columns:
        df["league"] = "E0"
    return df


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    """
    Load a single E0 CSV and return a clean DataFrame with:
      - Standardised column names
      - Parsed & sorted date
      - Derived targets: total_goals, over_2_5, over_1_5, over_3_5, btts
      - Normalised odds-implied probabilities (home / draw / away)
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig', on_bad_lines='skip')
    df = df.dropna(how='all').loc[:, lambda x: x.notna().any()]
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    if TEAM_ALIASES:
        df['home'] = df['home'].replace(TEAM_ALIASES)
        df['away'] = df['away'].replace(TEAM_ALIASES)
    df = _prepare_df(df)

    print(f"✅ Loaded {len(df)} matches from {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   Columns: {len(df.columns)}  |  "
          f"Results: H={df['result'].eq('H').sum()}  D={df['result'].eq('D').sum()}  A={df['result'].eq('A').sum()}")
    print(f"   Over 2.5: {df['over_2_5'].mean():.1%}  |  BTTS: {df['btts'].mean():.1%}")
    return df


# Big 5 league structure (from download.py)
BIG5_LEAGUES = ("E0", "D1", "SP1", "I1", "F1")
BIG5_SEASONS = ("1516", "1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526")

# Fallback column names for odds (some leagues use BbAv* instead of Avg*)
ODDS_FALLBACKS = [
    ("avg_odds_h", ["AvgH", "BbAvH"]),
    ("avg_odds_d", ["AvgD", "BbAvD"]),
    ("avg_odds_a", ["AvgA", "BbAvA"]),
    ("avg_odds_over25", ["Avg>2.5", "BbAv>2.5"]),
    ("avg_odds_under25", ["Avg<2.5", "BbAv<2.5"]),
]


def _load_single_csv(f: Path) -> pd.DataFrame:
    """Load one CSV, apply column mapping. League is added during concat."""
    df = pd.read_csv(f, encoding='utf-8-sig', on_bad_lines='skip')
    df = df.dropna(how='all').loc[:, lambda x: x.notna().any()]

    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    # Fallback odds columns (e.g. BbAvH when AvgH missing)
    for to_col, from_candidates in ODDS_FALLBACKS:
        if to_col not in rename.values():
            for src in from_candidates:
                if src in df.columns:
                    rename[src] = to_col
                    break
    return df.rename(columns=rename)


def load_all_seasons(data_dir: str, leagues: list[str] | None = None,
                     pattern: str = "pl_*.csv") -> tuple[pd.DataFrame, list[str]]:
    """
    Load match data from data_dir. Supports two structures:

    1. **Big 5 leagues**: data/{E0,D1,SP1,I1,F1}/{league}_{season}.csv
       - Set leagues=None to auto-detect and load all, or leagues=["E0","D1"] to filter.
    2. **Legacy**: data/pl_*.csv (Premier League only)

    Returns (DataFrame, list of file paths used).
    """
    data_path = Path(data_dir)
    files: list[Path] = []
    file_league: dict[Path, str] = {}

    # Try Big 5 structure first
    for league in (leagues or list(BIG5_LEAGUES)):
        league_dir = data_path / league
        if not league_dir.is_dir():
            continue
        for season in BIG5_SEASONS:
            f = league_dir / f"{league}_{season}.csv"
            if f.exists():
                files.append(f)
                file_league[f] = league

    if not files:
        # Fallback: legacy pl_*.csv
        files = sorted(data_path.glob(pattern))
        if not files:
            fallback = data_path / "pl_matches.csv"
            if fallback.exists():
                files = [fallback]
                print(f"📂 No Big 5 or {pattern} found, using {fallback.name}")
            else:
                raise FileNotFoundError(
                    f"No data in {data_dir}. Run download.py for Big 5, or add pl_*.csv"
                )

    dfs = []
    leagues_list = []
    for f in sorted(files):
        league = file_league.get(f) or "E0"
        dfs.append(_load_single_csv(f))
        leagues_list.append(league)

    combined = pd.concat(dfs, ignore_index=True).copy()
    # Add league column in one go
    combined["league"] = np.repeat(leagues_list, [len(d) for d in dfs])

    if TEAM_ALIASES:
        combined["home"] = combined["home"].replace(TEAM_ALIASES)
        combined["away"] = combined["away"].replace(TEAM_ALIASES)

    df = _prepare_df(combined)

    leagues_used = df["league"].unique().tolist() if "league" in df.columns else ["pl"]
    print(f"✅ Loaded {len(df)} matches from {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   Leagues: {leagues_used}  |  Files: {len(files)}")
    print(f"   Columns: {len(df.columns)}  |  "
          f"Results: H={df['result'].eq('H').sum()}  D={df['result'].eq('D').sum()}  A={df['result'].eq('A').sum()}")
    print(f"   Over 2.5: {df['over_2_5'].mean():.1%}  |  BTTS: {df['btts'].mean():.1%}")
    return df, [str(f) for f in files]


# ── Helpers for venue-neutral stats ────────────────────────────────────────────

def _overall_scored(frame: pd.DataFrame, team: str) -> float:
    """Average goals scored by `team` in `frame` (works for both home/away rows)."""
    if len(frame) == 0:
        return np.nan
    home_mask = (frame['home'] == team).values
    scored = np.where(home_mask, frame['home_goals'].values, frame['away_goals'].values)
    return float(np.nanmean(scored))


def _overall_conceded(frame: pd.DataFrame, team: str) -> float:
    """Average goals conceded by `team` in `frame`."""
    if len(frame) == 0:
        return np.nan
    home_mask = (frame['home'] == team).values
    conceded = np.where(home_mask, frame['away_goals'].values, frame['home_goals'].values)
    return float(np.nanmean(conceded))


def _overall_form_pts(frame: pd.DataFrame, team: str) -> float:
    """Sum of points earned by `team` in `frame` (3W / 1D / 0L)."""
    if len(frame) == 0:
        return np.nan
    pts = 0.0
    for _, r in frame.iterrows():
        res = r['result']
        if r['home'] == team:
            pts += 3.0 if res == 'H' else (1.0 if res == 'D' else 0.0)
        else:
            pts += 3.0 if res == 'A' else (1.0 if res == 'D' else 0.0)
    return pts


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Safe division; returns NaN if either input is NaN or denominator <= 0."""
    if np.isnan(numerator) or np.isnan(denominator) or denominator <= 0:
        return np.nan
    return numerator / denominator


def compute_elo_ratings(df: pd.DataFrame, k: float = 20.0,
                        home_adv: float = 100.0,
                        initial: float = 1500.0) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Compute Elo ratings for every match chronologically.
    Returns (DataFrame with home_elo/away_elo/elo_diff, latest elo dict).
    Ratings stored *before* the match result is applied (no leakage).
    """
    elos: dict[str, float] = {}
    home_elos, away_elos = [], []

    for _, row in df.iterrows():
        home, away = row['home'], row['away']
        h_elo = elos.get(home, initial)
        a_elo = elos.get(away, initial)

        home_elos.append(h_elo)
        away_elos.append(a_elo)

        exp_h = 1 / (1 + 10 ** ((a_elo - (h_elo + home_adv)) / 400))
        exp_a = 1 - exp_h

        actual_h = 1.0 if row['result'] == 'H' else (0.5 if row['result'] == 'D' else 0.0)
        actual_a = 1.0 - actual_h

        gd = abs(row['home_goals'] - row['away_goals'])
        gd_mult = max(1.0, np.log1p(gd))

        elos[home] = h_elo + k * gd_mult * (actual_h - exp_h)
        elos[away] = a_elo + k * gd_mult * (actual_a - exp_a)

    df = df.copy()
    df['home_elo'] = home_elos
    df['away_elo'] = away_elos
    df['elo_diff'] = df['home_elo'] - df['away_elo']
    return df, elos


def compute_team_stats(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    For each match, compute rolling team stats from the last `window` home/away games.
    Only uses past data to prevent leakage.
    """
    records = []

    for idx, row in df.iterrows():
        home, away, date = row['home'], row['away'], row['date']

        # Past home games for home team / past away games for away team
        past_home = df[(df['home'] == home) & (df['date'] < date)].tail(window)
        past_away = df[(df['away'] == away) & (df['date'] < date)].tail(window)

        # Overall recent games (all venues) for each team
        home_all_past = df[((df['home'] == home) | (df['away'] == home)) & (df['date'] < date)]
        away_all_past = df[((df['home'] == away) | (df['away'] == away)) & (df['date'] < date)]
        past_home_all = home_all_past.tail(window)
        past_away_all = away_all_past.tail(window)

        # Rest days since last game
        home_rest = float((date - home_all_past['date'].iloc[-1]).days) if len(home_all_past) > 0 else np.nan
        away_rest = float((date - away_all_past['date'].iloc[-1]).days) if len(away_all_past) > 0 else np.nan

        def avg(frame, col, default=np.nan):
            return frame[col].mean() if col in frame.columns and len(frame) > 0 else default

        def form_pts(frame, side):
            """Sum of points from last N games for given side (H or A)."""
            if len(frame) == 0 or 'result' not in frame.columns:
                return np.nan
            loss_val = 'A' if side == 'H' else 'H'
            pts = frame['result'].map({side: 3, 'D': 1, loss_val: 0})
            return pts.sum()

        h_goals_scored = avg(past_home, 'home_goals')
        h_goals_conceded = avg(past_home, 'away_goals')
        a_goals_scored = avg(past_away, 'away_goals')
        a_goals_conceded = avg(past_away, 'home_goals')

        record = {
            'idx': idx,
            # Goals
            'home_goals_scored_avg':   h_goals_scored,
            'home_goals_conceded_avg': h_goals_conceded,
            'away_goals_scored_avg':   a_goals_scored,
            'away_goals_conceded_avg': a_goals_conceded,
            # Goals diff (attacking vs defensive strength)
            'home_goals_diff': h_goals_scored - h_goals_conceded,
            'away_goals_diff': a_goals_scored - a_goals_conceded,
            # Shots
            'home_shots_avg':  avg(past_home, 'home_shots'),
            'home_sot_avg':    avg(past_home, 'home_sot'),
            'away_shots_avg':  avg(past_away, 'away_shots'),
            'away_sot_avg':    avg(past_away, 'away_sot'),
            # Corners
            'home_corners_avg': avg(past_home, 'home_corners'),
            'away_corners_avg': avg(past_away, 'away_corners'),
            # Fouls
            'home_fouls_avg': avg(past_home, 'home_fouls'),
            'away_fouls_avg': avg(past_away, 'away_fouls'),
            # Cards
            'home_yellows_avg': avg(past_home, 'home_yellows'),
            'away_yellows_avg': avg(past_away, 'away_yellows'),
            # Form
            'home_form': form_pts(past_home, 'H'),
            'away_form': form_pts(past_away, 'A'),
            # Overall (venue-neutral) strength
            'home_overall_goals_avg': _overall_scored(past_home_all, home),
            'home_overall_conceded_avg': _overall_conceded(past_home_all, home),
            'away_overall_goals_avg': _overall_scored(past_away_all, away),
            'away_overall_conceded_avg': _overall_conceded(past_away_all, away),
            'home_overall_form': _overall_form_pts(past_home_all, home),
            'away_overall_form': _overall_form_pts(past_away_all, away),
            # Shot efficiency
            'home_sot_ratio': _safe_ratio(avg(past_home, 'home_sot'), avg(past_home, 'home_shots')),
            'away_sot_ratio': _safe_ratio(avg(past_away, 'away_sot'), avg(past_away, 'away_shots')),
            # Rest
            'home_rest_days': home_rest,
            'away_rest_days': away_rest,
            # Odds features (from the current match row — not leakage, odds are pre-match)
            'implied_h':       row.get('implied_h', np.nan),
            'implied_d':       row.get('implied_d', np.nan),
            'implied_a':       row.get('implied_a', np.nan),
            'avg_implied_h':   row.get('avg_implied_h', np.nan),
            'avg_implied_d':   row.get('avg_implied_d', np.nan),
            'avg_implied_a':   row.get('avg_implied_a', np.nan),
            'implied_over25':  row.get('implied_over25', np.nan),
        }
        records.append(record)

    stats_df = pd.DataFrame(records).set_index('idx')
    # Drop columns already in df to avoid join collision
    overlap = [c for c in stats_df.columns if c in df.columns]
    stats_df = stats_df.drop(columns=overlap)
    return df.join(stats_df)


# ── Quick smoke test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else 'data'
    if os.path.isfile(arg):
        df = load_and_prepare(arg)
    else:
        df, _ = load_all_seasons(arg)
    df = compute_team_stats(df)
    print("\nFeature columns available for training:")
    feat_cols = [c for c in df.columns if any(c.endswith(s) for s in
                 ['_avg', '_form', 'implied_h', 'implied_d', 'implied_a',
                  'implied_over25', 'avg_implied_h', 'avg_implied_d', 'avg_implied_a'])]
    print(feat_cols)
    print(f"\nReady rows (no NaN in features): "
          f"{df.dropna(subset=feat_cols).shape[0]} / {len(df)}")