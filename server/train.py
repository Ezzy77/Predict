"""
train.py — Train XGBoost models for football prediction.
- Outcome (1X2): XGBClassifier
- Goals: separate home/away XGBRegressors (sum for total)
- Expanded features: implied odds, corners, yellows
- Time-series CV, better metrics, training medians, model metadata
"""

import os
import json
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
)
from scipy.stats import poisson
from xgboost import XGBClassifier, XGBRegressor

from data_prep import load_all_seasons, compute_team_stats, compute_elo_ratings


def _dc_tau(i: int, j: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Dixon-Coles tau correction."""
    if i == 0 and j == 0:
        return 1.0 - rho * lam_h * lam_a
    if i == 0 and j == 1:
        return 1.0 + rho * lam_h
    if i == 1 and j == 0:
        return 1.0 + rho * lam_a
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def _fit_dixon_coles_rho(
    home_exp: np.ndarray,
    away_exp: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    rho_grid: np.ndarray | None = None,
) -> float:
    """Fit rho by maximizing log-likelihood of observed scorelines under Dixon-Coles."""
    if rho_grid is None:
        rho_grid = np.linspace(-0.25, 0.0, 26)
    best_rho, best_ll = -0.13, -np.inf
    for rho in rho_grid:
        ll = 0.0
        for h_exp, a_exp, h_obs, a_obs in zip(home_exp, away_exp, home_goals, away_goals):
            h_obs, a_obs = int(min(h_obs, 9)), int(min(a_obs, 9))
            p_h = poisson.pmf(h_obs, max(0.01, h_exp))
            p_a = poisson.pmf(a_obs, max(0.01, a_exp))
            tau = _dc_tau(h_obs, a_obs, h_exp, a_exp, rho)
            prob = tau * p_h * p_a
            ll += np.log(max(prob, 1e-10))
        if ll > best_ll:
            best_ll, best_rho = ll, rho
    return float(best_rho)

# League encoding for multi-league model
LEAGUE_TO_ID = {"E0": 0, "D1": 1, "SP1": 2, "I1": 3, "F1": 4}

# ── Expanded feature set (includes implied odds, corners, yellows, fouls, goals_diff) ─
FEATURE_COLS = [
    'home_goals_scored_avg', 'home_goals_conceded_avg', 'home_goals_diff',
    'home_shots_avg', 'home_sot_avg', 'home_corners_avg', 'home_fouls_avg', 'home_yellows_avg',
    'home_form',
    'away_goals_scored_avg', 'away_goals_conceded_avg', 'away_goals_diff',
    'away_shots_avg', 'away_sot_avg', 'away_corners_avg', 'away_fouls_avg', 'away_yellows_avg',
    'away_form',
    'implied_h', 'implied_d', 'implied_a',
    'avg_implied_h', 'avg_implied_d', 'avg_implied_a',
    'implied_over25',
    # ── New features ──
    'home_overall_goals_avg', 'home_overall_conceded_avg',
    'away_overall_goals_avg', 'away_overall_conceded_avg',
    'home_overall_form', 'away_overall_form',
    'home_sot_ratio', 'away_sot_ratio',
    'home_rest_days', 'away_rest_days',
    'home_elo', 'away_elo', 'elo_diff',
    'league_id',
]


def _fill_missing_with_median(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Fill NaN with column medians. Returns (filled df, median dict for inference)."""
    medians = {}
    for c in cols:
        if c in df.columns:
            med = df[c].median()
            medians[c] = float(med) if not np.isnan(med) else 0.0
    df_filled = df.copy()
    for c, v in medians.items():
        df_filled[c] = df_filled[c].fillna(v)
    return df_filled, medians


def _time_series_cv(
    X: pd.DataFrame,
    y_result: np.ndarray,
    y_home: pd.Series,
    y_away: pd.Series,
    n_splits: int = 5,
) -> dict:
    """Run time-series cross-validation and return metrics."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    outcome_acc, outcome_logloss = [], []
    goals_mae, goals_rmse = [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_res_train = y_result[train_idx]
        y_res_test = y_result[test_idx]
        y_h_train = y_home.iloc[train_idx]
        y_h_test = y_home.iloc[test_idx]
        y_a_train = y_away.iloc[train_idx]
        y_a_test = y_away.iloc[test_idx]

        clf = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.03,
                            eval_metric='mlogloss', early_stopping_rounds=30,
                            subsample=0.8, colsample_bytree=0.8,
                            min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0)
        sw = compute_sample_weight('balanced', y_res_train)
        clf.fit(X_train, y_res_train, sample_weight=sw,
                eval_set=[(X_test, y_res_test)], verbose=False)
        preds = clf.predict(X_test)
        outcome_acc.append(accuracy_score(y_res_test, preds))
        try:
            outcome_logloss.append(log_loss(y_res_test, clf.predict_proba(X_test)))
        except Exception:
            outcome_logloss.append(np.nan)

        reg_h = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.03,
                             early_stopping_rounds=30, subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0)
        reg_a = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.03,
                             early_stopping_rounds=30, subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0)
        reg_h.fit(X_train, y_h_train, eval_set=[(X_test, y_h_test)], verbose=False)
        reg_a.fit(X_train, y_a_train, eval_set=[(X_test, y_a_test)], verbose=False)
        pred_h = np.maximum(0, reg_h.predict(X_test))
        pred_a = np.maximum(0, reg_a.predict(X_test))
        total_pred = pred_h + pred_a
        total_actual = y_h_test.values + y_a_test.values
        goals_mae.append(mean_absolute_error(total_actual, total_pred))
        goals_rmse.append(np.sqrt(mean_squared_error(total_actual, total_pred)))

    return {
        'outcome_accuracy_mean': float(np.mean(outcome_acc)),
        'outcome_accuracy_std': float(np.std(outcome_acc)),
        'outcome_logloss_mean': float(np.nanmean(outcome_logloss)),
        'goals_mae_mean': float(np.mean(goals_mae)),
        'goals_mae_std': float(np.std(goals_mae)),
        'goals_rmse_mean': float(np.mean(goals_rmse)),
    }


# Set to True to run hyperparameter tuning (slower)
TUNE_HYPERPARAMS = False

# Grid for outcome model tuning
OUTCOME_PARAM_GRID = {
    'n_estimators': [200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1],
}


def train(data_dir: str = "data"):
    os.makedirs('models', exist_ok=True)

    df, files_used = load_all_seasons(data_dir)
    df["league_id"] = df["league"].map(LEAGUE_TO_ID).fillna(0).astype(int)
    df = compute_team_stats(df)

    # Compute Elo ratings
    df, _elo_latest = compute_elo_ratings(df)
    print(f"   Elo range: {df['home_elo'].min():.0f} – {df['home_elo'].max():.0f}")

    # Ensure all feature columns exist; fill missing with 0 for optional cols
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0

    # Require core stats; implied odds can be NaN (filled by medians)
    core = ['home_goals_scored_avg', 'away_goals_scored_avg', 'result', 'total_goals', 'home_goals', 'away_goals']
    df = df.dropna(subset=[c for c in core if c in df.columns])

    X = df[FEATURE_COLS].copy()
    X, training_medians = _fill_missing_with_median(X, FEATURE_COLS)

    le = LabelEncoder()
    y_result = le.fit_transform(df['result'].values)
    y_goals = df['total_goals'].reset_index(drop=True)
    y_home = df['home_goals'].reset_index(drop=True)
    y_away = df['away_goals'].reset_index(drop=True)

    # Time-series split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_res_train = y_result[:split_idx]
    y_res_test = y_result[split_idx:]
    y_home_train = y_home.iloc[:split_idx]
    y_home_test = y_home.iloc[split_idx:]
    y_away_train = y_away.iloc[:split_idx]
    y_away_test = y_away.iloc[split_idx:]
    y_goal_train = y_goals.iloc[:split_idx]
    y_goal_test = y_goals.iloc[split_idx:]

    # Time-series CV
    print("Running time-series cross-validation...")
    cv_metrics = _time_series_cv(X, y_result, y_home, y_away, n_splits=5)
    for k, v in cv_metrics.items():
        print(f"  CV {k}: {v:.4f}")

    # --- Model A: Match Outcome ---
    clf = XGBClassifier(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.02,
        eval_metric='mlogloss',
        early_stopping_rounds=50,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
    )
    if TUNE_HYPERPARAMS:
        print("\nTuning outcome model hyperparameters...")
        tscv = TimeSeriesSplit(n_splits=5)
        grid = GridSearchCV(
            XGBClassifier(eval_metric='mlogloss'),
            OUTCOME_PARAM_GRID,
            cv=tscv,
            scoring='neg_log_loss',
            n_jobs=-1,
            verbose=1,
        )
        sample_weight = compute_sample_weight('balanced', y_res_train)
        grid.fit(X_train, y_res_train, sample_weight=sample_weight)
        clf = grid.best_estimator_
        print(f"  Best params: {grid.best_params_} | CV log loss: {-grid.best_score_:.4f}")
    else:
        sample_weight = compute_sample_weight('balanced', y_res_train)
        clf.fit(X_train, y_res_train, sample_weight=sample_weight,
                eval_set=[(X_test, y_res_test)], verbose=False)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_res_test, preds)
    ll = log_loss(y_res_test, clf.predict_proba(X_test))
    print(f"\nOutcome — Accuracy: {acc:.3f} | Log loss: {ll:.4f}")
    print("Confusion matrix:")

    cm = confusion_matrix(y_res_test, preds)
    print(cm)

    # --- Model B: Home Goals ---
    reg_home = XGBRegressor(
        n_estimators=1000, max_depth=4, learning_rate=0.02,
        early_stopping_rounds=50, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
    )
    reg_home.fit(X_train, y_home_train,
                 eval_set=[(X_test, y_home_test)], verbose=False)
    pred_home = np.maximum(0, reg_home.predict(X_test))

    # --- Model C: Away Goals ---
    reg_away = XGBRegressor(
        n_estimators=1000, max_depth=4, learning_rate=0.02,
        early_stopping_rounds=50, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
    )
    reg_away.fit(X_train, y_away_train,
                 eval_set=[(X_test, y_away_test)], verbose=False)
    pred_away = np.maximum(0, reg_away.predict(X_test))

    total_pred = pred_home + pred_away
    mae = mean_absolute_error(y_goal_test, total_pred)
    rmse = np.sqrt(mean_squared_error(y_goal_test, total_pred))
    r2 = 1 - (np.sum((y_goal_test - total_pred) ** 2) / np.sum((y_goal_test - y_goal_test.mean()) ** 2))
    print(f"\nGoals (home+away) — MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f}")

    # --- Fit Dixon-Coles rho from training data ---
    pred_home_all = np.maximum(0, reg_home.predict(X_train))
    pred_away_all = np.maximum(0, reg_away.predict(X_train))
    dc_rho = _fit_dixon_coles_rho(
        pred_home_all, pred_away_all,
        y_home_train.values, y_away_train.values,
    )
    print(f"Dixon-Coles rho (fitted): {dc_rho:.4f}")

    # --- Save artifacts ---
    joblib.dump(clf, 'models/outcome_model.pkl')
    joblib.dump(reg_home, 'models/home_goals_model.pkl')
    joblib.dump(reg_away, 'models/away_goals_model.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    joblib.dump(FEATURE_COLS, 'models/feature_cols.pkl')
    joblib.dump(training_medians, 'models/training_medians.pkl')
    joblib.dump(dc_rho, 'models/dixon_coles_rho.pkl')

    metadata = {
        'trained_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'data_path': data_dir,
        'data_files': files_used,
        'n_samples': len(df),
        'outcome_accuracy': float(acc),
        'outcome_log_loss': float(ll),
        'goals_mae': float(mae),
        'goals_rmse': float(rmse),
        'goals_r2': float(r2),
        'dixon_coles_rho': float(dc_rho),
        'cv_metrics': cv_metrics,
    }
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n✅ Models saved to models/")
    print(f"   Metadata: {metadata['trained_at']}")


if __name__ == '__main__':
    train()
