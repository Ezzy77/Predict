"""
api/main.py — Football Prediction REST API
Endpoints:
  POST /predict          — predict from raw pre-computed features
  POST /predict/fixture  — predict from team names + optional odds (looks up history automatically)
  GET  /teams            — list all known teams
  GET  /health           — health check
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from scipy.stats import poisson
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date

warnings.filterwarnings("ignore")

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="⚽ Football Prediction API",
    description="Predict match outcomes (1X2) and goals (over/under) for Premier League fixtures.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models & historical data at startup ──────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_artifact(filename):
    path = os.path.join(BASE_DIR, "models", filename)
    if not os.path.exists(path):
        raise RuntimeError(f"Model file not found: {path}. Run train.py first.")
    return joblib.load(path)

outcome_model = load_artifact("outcome_model.pkl")
label_encoder = load_artifact("label_encoder.pkl")
feature_cols = load_artifact("feature_cols.pkl")

try:
    training_medians = load_artifact("training_medians.pkl")
except RuntimeError:
    training_medians = {}

try:
    home_goals_model = load_artifact("home_goals_model.pkl")
    away_goals_model = load_artifact("away_goals_model.pkl")
    goals_model = None
except RuntimeError:
    home_goals_model = away_goals_model = None
    goals_model = load_artifact("goals_model.pkl")

# Load historical match data for rolling stat lookups
DATA_DIR = os.path.join(BASE_DIR, "data")

def load_history() -> pd.DataFrame:
    """Load + prepare historical data for rolling lookups (multi-season)."""
    from data_prep import load_all_seasons, compute_elo_ratings
    df, _ = load_all_seasons(DATA_DIR)
    df, elos = compute_elo_ratings(df)
    return df, elos

try:
    history_df, team_elos = load_history()
    KNOWN_TEAMS = sorted(set(history_df['home'].dropna()) | set(history_df['away'].dropna()))
    print(f"✅ Loaded {len(history_df)} historical matches. {len(KNOWN_TEAMS)} teams known.")
except Exception as e:
    history_df = pd.DataFrame()
    KNOWN_TEAMS = []
    team_elos = {}
    print(f"⚠️  Could not load historical data: {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rolling_team_stats(home: str, away: str,
                        as_of: pd.Timestamp, window: int = 5,
                        league: str | None = None) -> dict:
    """
    Compute rolling stats for home and away team using the last `window`
    home/away games played before `as_of`.
    Returns a flat dict of feature values.
    """
    df = history_df
    if league and "league" in df.columns:
        df = df[df["league"] == league]

    past_home = df[(df['home'] == home) & (df['date'] < as_of)].tail(window)
    past_away = df[(df['away'] == away) & (df['date'] < as_of)].tail(window)

    # Overall recent games (all venues)
    home_all_past = df[((df['home'] == home) | (df['away'] == home)) & (df['date'] < as_of)]
    away_all_past = df[((df['home'] == away) | (df['away'] == away)) & (df['date'] < as_of)]
    past_home_all = home_all_past.tail(window)
    past_away_all = away_all_past.tail(window)

    def avg(frame, col):
        return float(frame[col].mean()) if col in frame.columns and len(frame) > 0 else np.nan

    def form_pts(frame, side):
        if len(frame) == 0 or 'result' not in frame.columns:
            return np.nan
        loss = 'A' if side == 'H' else 'H'
        return float(frame['result'].map({side: 3, 'D': 1, loss: 0}).sum())

    def overall_scored(frame, team):
        if len(frame) == 0:
            return np.nan
        hm = (frame['home'] == team).values
        scored = np.where(hm, frame['home_goals'].values, frame['away_goals'].values)
        return float(np.nanmean(scored))

    def overall_conceded(frame, team):
        if len(frame) == 0:
            return np.nan
        hm = (frame['home'] == team).values
        conc = np.where(hm, frame['away_goals'].values, frame['home_goals'].values)
        return float(np.nanmean(conc))

    def overall_form(frame, team):
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

    def safe_ratio(num, den):
        if np.isnan(num) or np.isnan(den) or den <= 0:
            return np.nan
        return num / den

    h_goals_scored = avg(past_home, 'home_goals')
    h_goals_conceded = avg(past_home, 'away_goals')
    a_goals_scored = avg(past_away, 'away_goals')
    a_goals_conceded = avg(past_away, 'home_goals')

    # Rest days
    home_rest = float((as_of - home_all_past['date'].iloc[-1]).days) if len(home_all_past) > 0 else np.nan
    away_rest = float((as_of - away_all_past['date'].iloc[-1]).days) if len(away_all_past) > 0 else np.nan

    return {
        'home_goals_scored_avg':   h_goals_scored,
        'home_goals_conceded_avg': h_goals_conceded,
        'home_goals_diff':         h_goals_scored - h_goals_conceded,
        'away_goals_scored_avg':   a_goals_scored,
        'away_goals_conceded_avg': a_goals_conceded,
        'away_goals_diff':         a_goals_scored - a_goals_conceded,
        'home_shots_avg':          avg(past_home, 'home_shots'),
        'home_sot_avg':            avg(past_home, 'home_sot'),
        'away_shots_avg':          avg(past_away, 'away_shots'),
        'away_sot_avg':            avg(past_away, 'away_sot'),
        'home_corners_avg':        avg(past_home, 'home_corners'),
        'away_corners_avg':        avg(past_away, 'away_corners'),
        'home_fouls_avg':          avg(past_home, 'home_fouls'),
        'away_fouls_avg':          avg(past_away, 'away_fouls'),
        'home_yellows_avg':        avg(past_home, 'home_yellows'),
        'away_yellows_avg':        avg(past_away, 'away_yellows'),
        'home_form':               form_pts(past_home, 'H'),
        'away_form':               form_pts(past_away, 'A'),
        # Overall (venue-neutral)
        'home_overall_goals_avg':     overall_scored(past_home_all, home),
        'home_overall_conceded_avg':  overall_conceded(past_home_all, home),
        'away_overall_goals_avg':     overall_scored(past_away_all, away),
        'away_overall_conceded_avg':  overall_conceded(past_away_all, away),
        'home_overall_form':          overall_form(past_home_all, home),
        'away_overall_form':          overall_form(past_away_all, away),
        # Shot efficiency
        'home_sot_ratio': safe_ratio(avg(past_home, 'home_sot'), avg(past_home, 'home_shots')),
        'away_sot_ratio': safe_ratio(avg(past_away, 'away_sot'), avg(past_away, 'away_shots')),
        # Rest days
        'home_rest_days': home_rest,
        'away_rest_days': away_rest,
        # Elo
        'home_elo': team_elos.get(home, 1500.0),
        'away_elo': team_elos.get(away, 1500.0),
        'elo_diff': team_elos.get(home, 1500.0) - team_elos.get(away, 1500.0),
        # League
        'league_id': LEAGUE_TO_ID.get(league or "E0", 0),
        # Metadata (popped before model input)
        '_home_games_found':       len(past_home),
        '_away_games_found':       len(past_away),
    }


def _odds_to_implied(odds_h: Optional[float],
                     odds_d: Optional[float],
                     odds_a: Optional[float],
                     odds_over25: Optional[float] = None) -> dict:
    """Convert raw decimal odds → margin-normalised implied probabilities."""
    result = {
        'implied_h': np.nan, 'implied_d': np.nan, 'implied_a': np.nan,
        'avg_implied_h': np.nan, 'avg_implied_d': np.nan, 'avg_implied_a': np.nan,
        'implied_over25': np.nan,
    }
    if all(v is not None and v > 1 for v in [odds_h, odds_d, odds_a]):
        overround = (1/odds_h) + (1/odds_d) + (1/odds_a)
        result['implied_h'] = round((1/odds_h) / overround, 4)
        result['implied_d'] = round((1/odds_d) / overround, 4)
        result['implied_a'] = round((1/odds_a) / overround, 4)
        # also fill avg_implied with same values if no separate avg odds given
        result['avg_implied_h'] = result['implied_h']
        result['avg_implied_d'] = result['implied_d']
        result['avg_implied_a'] = result['implied_a']
    if odds_over25 is not None and odds_over25 > 1:
        # We don't have under odds here — use a simple inversion approximation
        result['implied_over25'] = round(1 / odds_over25, 4)
    return result


def _run_models(features: dict) -> dict:
    """Feed feature dict into both models and return full prediction dict."""
    X = pd.DataFrame([features])
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feature_cols]

    # Fill NaNs with training medians (stored during train.py); fallback to 0
    for c in feature_cols:
        val = X[c].iloc[0]
        if pd.isna(val) or (isinstance(val, float) and np.isnan(val)):
            X[c] = training_medians.get(c, 0.0)

    # Outcome probabilities
    proba = outcome_model.predict_proba(X)[0]
    classes = label_encoder.classes_
    prob_map = dict(zip(classes, proba))
    predicted = label_encoder.inverse_transform([outcome_model.predict(X)[0]])[0]

    # Goals: separate home/away if available, else total
    if home_goals_model is not None and away_goals_model is not None:
        home_exp = float(np.maximum(0, home_goals_model.predict(X)[0]))
        away_exp = float(np.maximum(0, away_goals_model.predict(X)[0]))
        expected_goals = home_exp + away_exp
    else:
        expected_goals = float(np.maximum(0, goals_model.predict(X)[0]))
        h_prob = prob_map.get('H', 0.33)
        a_prob = prob_map.get('A', 0.33)
        total_prob = h_prob + a_prob or 1
        home_exp = expected_goals * (h_prob / total_prob)
        away_exp = expected_goals - home_exp

    # Over/under via Poisson on total
    over_25 = float(1 - poisson.cdf(2, expected_goals))
    over_15 = float(1 - poisson.cdf(1, expected_goals))
    over_35 = float(1 - poisson.cdf(3, expected_goals))

    outcome_label = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[predicted]

    return {
        "predicted_outcome": predicted,
        "outcome_label": outcome_label,
        "home_win_prob": round(float(prob_map.get('H', 0)), 4),
        "draw_prob": round(float(prob_map.get('D', 0)), 4),
        "away_win_prob": round(float(prob_map.get('A', 0)), 4),
        "expected_goals": round(expected_goals, 2),
        "home_expected_goals": round(home_exp, 2),
        "away_expected_goals": round(away_exp, 2),
        "over_1_5_prob": round(over_15, 4),
        "over_2_5_prob": round(over_25, 4),
        "under_2_5_prob": round(1 - over_25, 4),
        "over_3_5_prob": round(over_35, 4),
    }


# ── Schemas ───────────────────────────────────────────────────────────────────

class RawFeatures(BaseModel):
    """Manually supply every pre-computed feature value."""
    home_goals_scored_avg:   float
    home_goals_conceded_avg: float
    home_goals_diff:         Optional[float] = None
    home_shots_avg:          float
    home_sot_avg:            float
    home_corners_avg:        float = 0.0
    home_fouls_avg:          Optional[float] = None
    home_yellows_avg:        float = 0.0
    home_form:               float        # points from last 5 home games (0-15)
    away_goals_scored_avg:   float
    away_goals_conceded_avg: float
    away_goals_diff:         Optional[float] = None
    away_shots_avg:          float
    away_sot_avg:            float
    away_corners_avg:        float = 0.0
    away_fouls_avg:          Optional[float] = None
    away_yellows_avg:        float = 0.0
    away_form:               float        # points from last 5 away games (0-15)
    implied_h:               Optional[float] = None
    implied_d:               Optional[float] = None
    implied_a:               Optional[float] = None
    avg_implied_h:           Optional[float] = None
    avg_implied_d:           Optional[float] = None
    avg_implied_a:           Optional[float] = None
    implied_over25:          Optional[float] = None


LEAGUE_TO_ID = {"E0": 0, "D1": 1, "SP1": 2, "I1": 3, "F1": 4}
LEAGUE_NAMES = {"E0": "Premier League", "D1": "Bundesliga", "SP1": "La Liga", "I1": "Serie A", "F1": "Ligue 1"}


class FixtureRequest(BaseModel):
    """Just provide team names — stats are looked up from history automatically."""
    home_team:    str = Field(..., example="Arsenal")
    away_team:    str = Field(..., example="Chelsea")
    league:       Optional[str] = Field(default="E0", example="E0",
                                        description="League code: E0, D1, SP1, I1, F1")
    match_date:   Optional[date] = Field(
        default=None,
        description="Date of the fixture (YYYY-MM-DD). Defaults to today if omitted."
    )
    # Optional: supply odds to improve prediction
    odds_home:    Optional[float] = Field(default=None, example=2.1,
                                          description="Decimal odds for home win (e.g. B365)")
    odds_draw:    Optional[float] = Field(default=None, example=3.4)
    odds_away:    Optional[float] = Field(default=None, example=3.6)
    odds_over25:  Optional[float] = Field(default=None, example=1.85,
                                          description="Decimal odds for over 2.5 goals")
    window:       int = Field(default=5, ge=1, le=20,
                              description="Number of recent games to average over")


class PredictionResponse(BaseModel):
    predicted_outcome: str
    outcome_label: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    expected_goals: float
    home_expected_goals: float = 0.0
    away_expected_goals: float = 0.0
    over_1_5_prob: float
    over_2_5_prob: float
    under_2_5_prob: float
    over_3_5_prob: float


class FixturePredictionResponse(PredictionResponse):
    home_team:         str
    away_team:         str
    match_date:        str
    home_games_used:   int
    away_games_used:   int
    odds_provided:     bool
    features_used:     dict


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    return {
        "message": "⚽ Football Prediction API",
        "version": "1.1.0",
        "docs": "/docs",
        "endpoints": ["/predict", "/predict/fixture", "/teams", "/health"],
    }


@app.get("/health", tags=["General"])
def health():
    meta = {}
    meta_path = os.path.join(BASE_DIR, "models", "metadata.json")
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            meta = json.load(f)
    return {
        "status": "ok",
        "models_loaded": True,
        "historical_matches": len(history_df),
        "known_teams": len(KNOWN_TEAMS),
        "model_version": meta.get("trained_at", None),
    }


@app.get("/models/metadata", tags=["General"])
def models_metadata():
    """Return training metadata (metrics, date) when available."""
    meta_path = os.path.join(BASE_DIR, "models", "metadata.json")
    if not os.path.exists(meta_path):
        return {"message": "No metadata (run train.py to generate)"}
    import json
    with open(meta_path) as f:
        return json.load(f)


@app.get("/teams", tags=["General"])
def list_teams(league: Optional[str] = None):
    """Return team names from the training data. Optionally filter by league (E0, D1, SP1, I1, F1)."""
    if league and "league" in history_df.columns:
        sub = history_df[history_df["league"] == league]
        league_teams = sorted(set(sub["home"].dropna()) | set(sub["away"].dropna()))
        teams = league_teams
    else:
        teams = KNOWN_TEAMS
    return {"teams": teams, "count": len(teams)}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_from_features(features: RawFeatures):
    """
    Predict match outcome and goals from manually supplied feature values.
    Use this if you are computing rolling stats yourself.
    """
    feat_dict = features.model_dump()
    # Fill implied probs with NaN if not provided
    for k in ['implied_h','implied_d','implied_a',
              'avg_implied_h','avg_implied_d','avg_implied_a','implied_over25']:
        if feat_dict.get(k) is None:
            feat_dict[k] = np.nan
    try:
        return _run_models(feat_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/fixture", response_model=FixturePredictionResponse, tags=["Prediction"])
def predict_fixture(req: FixtureRequest):
    """
    Predict a match by providing only team names and (optionally) odds.

    The API automatically looks up each team's last N home/away games from
    historical data and computes the rolling feature averages.

    - **home_team** / **away_team**: must match names in /teams exactly
    - **match_date**: used as the cut-off for historical lookups (defaults to today)
    - **odds_home/draw/away**: supply bookmaker odds to boost accuracy
    - **window**: how many recent games to average (default 5)
    """
    if history_df.empty:
        raise HTTPException(
            status_code=503,
            detail="Historical data not loaded. Check that data/ contains pl_*.csv or pl_matches.csv."
        )

    # Validate team names (case-insensitive fuzzy match)
    def resolve_team(name: str) -> str:
        if name in KNOWN_TEAMS:
            return name
        match = next((t for t in KNOWN_TEAMS if t.lower() == name.lower()), None)
        if match:
            return match
        close = [t for t in KNOWN_TEAMS if name.lower() in t.lower()]
        if len(close) == 1:
            return close[0]
        raise HTTPException(
            status_code=422,
            detail=f"Unknown team '{name}'. "
                   f"{'Did you mean: ' + ', '.join(close) + '?' if close else ''}"
                   f" See /teams for the full list."
        )

    home_team = resolve_team(req.home_team)
    away_team = resolve_team(req.away_team)

    if home_team == away_team:
        raise HTTPException(status_code=422, detail="Home and away teams must be different.")

    # Resolve match date
    as_of = pd.Timestamp(req.match_date) if req.match_date else pd.Timestamp.today()

    # Build rolling stats
    stats = _rolling_team_stats(home_team, away_team, as_of, window=req.window, league=req.league)
    home_games_used = int(stats.pop('_home_games_found'))
    away_games_used = int(stats.pop('_away_games_found'))

    if home_games_used == 0:
        raise HTTPException(
            status_code=422,
            detail=f"No historical home games found for '{home_team}' before {as_of.date()}."
        )
    if away_games_used == 0:
        raise HTTPException(
            status_code=422,
            detail=f"No historical away games found for '{away_team}' before {as_of.date()}."
        )

    # Build odds-implied features
    odds_implied = _odds_to_implied(
        req.odds_home, req.odds_draw, req.odds_away, req.odds_over25
    )
    odds_provided = req.odds_home is not None

    # Merge all features
    features = {**stats, **odds_implied}

    try:
        prediction = _run_models(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return FixturePredictionResponse(
        **prediction,
        home_team=home_team,
        away_team=away_team,
        match_date=str(as_of.date()),
        home_games_used=home_games_used,
        away_games_used=away_games_used,
        odds_provided=odds_provided,
        features_used={k: round(v, 4) if isinstance(v, float) and not np.isnan(v) else v
                       for k, v in features.items()},
    )


@app.post("/predict/fixture/batch", tags=["Prediction"])
def predict_fixture_batch(fixtures: list[FixtureRequest]):
    """Predict multiple fixtures at once. Max 50 per request."""
    if len(fixtures) > 50:
        raise HTTPException(status_code=422, detail="Max 50 fixtures per batch request.")
    return [predict_fixture(f) for f in fixtures]
