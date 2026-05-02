"""
ML Engine for Ad Performance Prediction.

Trains multiple ML algorithms on a synthetic ad-performance dataset,
selects the best model via cross-validation, and exposes a simple
predict() interface for the Flask app.
"""

import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# ---------------------------------------------------------------------------
# Synthetic training dataset (~200 samples)
# Features: [headline_length, ad_type, color_score, audience, platform,
#             posting_hour, posting_day]
#
#   ad_type:  1=Poster, 2=Social, 3=Banner, 4=MicroWeb, 5=Email
#   audience: 1=GenZ/Students, 2=B2B, 3=General
#   platform: 1=Instagram, 2=YouTube, 3=Facebook
#   posting_day: 1=Mon .. 7=Sun
# ---------------------------------------------------------------------------

_rng = np.random.RandomState(42)

def _generate_dataset(n: int = 250) -> tuple[np.ndarray, np.ndarray]:
    """Generate a reproducible synthetic dataset with realistic patterns."""
    headline_lengths = _rng.randint(3, 30, size=n)
    ad_types = _rng.choice([1, 2, 3, 4, 5], size=n)
    color_scores = _rng.randint(1, 11, size=n)
    audiences = _rng.choice([1, 2, 3], size=n)
    platforms = _rng.choice([1, 2, 3], size=n)
    posting_hours = _rng.randint(0, 24, size=n)
    posting_days = _rng.randint(1, 8, size=n)

    X = np.column_stack([
        headline_lengths, ad_types, color_scores,
        audiences, platforms, posting_hours, posting_days,
    ])

    # Build target with realistic heuristics + noise
    scores = np.full(n, 55.0)

    # Headline length: sweet spot 6-15 words
    optimal_hl = np.where(
        (headline_lengths >= 6) & (headline_lengths <= 15), 12.0, -8.0
    )
    scores += optimal_hl

    # Color vibrancy
    scores += (color_scores - 5) * 2.5

    # Video/social ads perform well
    scores += np.where(ad_types == 2, 8.0, 0.0)

    # Platform-audience fit
    scores += np.where((audiences == 1) & (platforms == 1), 10.0, 0.0)
    scores += np.where((audiences == 2) & (platforms == 3), 8.0, 0.0)
    scores -= np.where((audiences == 2) & (platforms == 1), 10.0, 0.0)

    # Peak posting hours (18-21)
    peak = np.where((posting_hours >= 18) & (posting_hours <= 21), 10.0, 0.0)
    off_peak = np.where((posting_hours < 7) | ((posting_hours > 13) & (posting_hours < 17)), -6.0, 0.0)
    scores += peak + off_peak

    # Weekend boost for B2C, penalty for B2B
    weekend = (posting_days >= 6).astype(float)
    scores += np.where(audiences != 2, weekend * 8, -weekend * 12)

    # Gaussian noise
    scores += _rng.normal(0, 3, size=n)
    scores = np.clip(scores, 0, 100).astype(int)

    return X, scores


def _build_candidates() -> dict[str, Pipeline]:
    """Return named candidate pipelines."""
    return {
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=120, max_depth=8, random_state=42,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=150, max_depth=4, learning_rate=0.1,
                random_state=42,
            )),
        ]),
        "svr": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=50, epsilon=2)),
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=7, weights="distance")),
        ]),
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=2000,
                early_stopping=True,
                random_state=42,
            )),
        ]),
    }


class AdScorePredictor:
    """Trains multiple models, picks the best, and predicts ad scores."""

    def __init__(self) -> None:
        self.X, self.y = _generate_dataset()
        self.best_model_name: str = ""
        self.cv_results: dict[str, float] = {}
        self.model: Pipeline | VotingRegressor = self._train()

    def _train(self) -> Pipeline | VotingRegressor:
        candidates = _build_candidates()
        best_score = -np.inf
        best_name = ""

        for name, pipeline in candidates.items():
            cv = cross_val_score(
                pipeline, self.X, self.y, cv=5, scoring="neg_mean_absolute_error",
            )
            mean_score = cv.mean()
            self.cv_results[name] = round(-mean_score, 2)
            if mean_score > best_score:
                best_score = mean_score
                best_name = name

        self.best_model_name = best_name

        # Build a voting ensemble of the top-3 models for robustness
        sorted_names = sorted(self.cv_results, key=self.cv_results.get)
        top3 = sorted_names[:3]
        estimators = [(n, candidates[n]) for n in top3]

        ensemble = VotingRegressor(estimators=estimators)
        ensemble.fit(self.X, self.y)
        return ensemble

    def predict(
        self,
        headline_length: int,
        ad_type: int,
        color_score: int,
        audience: int,
        platform: int,
        posting_hour: int,
        posting_day: int,
    ) -> int:
        features = np.array([[
            headline_length, ad_type, color_score,
            audience, platform, posting_hour, posting_day,
        ]])
        raw = self.model.predict(features)[0]
        return int(np.clip(raw, 0, 100))


# Module-level singleton so the model is trained once at import time
predictor = AdScorePredictor()
