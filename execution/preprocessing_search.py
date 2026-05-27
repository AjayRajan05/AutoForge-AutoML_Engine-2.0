"""
Preprocessing recipe search — select a transform recipe before model search.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold

from execution.preprocessing_pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)

DEFAULT_RECIPES: Dict[str, Dict[str, Any]] = {
    "standard": {
        "scale_features": True,
        "scale_type": "standard",
        "impute_strategy": "median",
        "encoding_strategy": "ordinal",
        "feature_engineering": False,
    },
    "robust": {
        "scale_features": True,
        "scale_type": "robust",
        "impute_strategy": "median",
        "encoding_strategy": "ordinal",
        "feature_engineering": False,
    },
    "minimal": {
        "scale_features": False,
        "scale_type": "none",
        "impute_strategy": "median",
        "encoding_strategy": "ordinal",
        "feature_engineering": False,
    },
    "poly": {
        "scale_features": True,
        "scale_type": "standard",
        "impute_strategy": "median",
        "encoding_strategy": "ordinal",
        "feature_engineering": True,
        "polynomial_degree": 2,
    },
    "onehot_low_card": {
        "scale_features": True,
        "scale_type": "standard",
        "impute_strategy": "median",
        "encoding_strategy": "onehot",
        "max_categories": 10,
        "feature_engineering": False,
    },
    "quantile": {
        "scale_features": True,
        "scale_type": "quantile",
        "impute_strategy": "median",
        "encoding_strategy": "ordinal",
        "feature_engineering": False,
    },
    "no_scale_tree": {
        "scale_features": False,
        "scale_type": "none",
        "impute_strategy": "median",
        "encoding_strategy": "ordinal",
        "feature_engineering": False,
    },
    "target_encode": {
        "scale_features": False,
        "scale_type": "none",
        "impute_strategy": "median",
        "encoding_strategy": "target",
        "feature_engineering": False,
    },
}


def default_scoring(task_type: str) -> str:
    """Default sklearn scoring string per task type."""
    return "accuracy" if task_type == "classification" else "neg_mean_squared_error"


class PreprocessingSearchSpace:
    """Evaluate preprocessing recipes with fast screen models inside CV."""

    def __init__(
        self,
        recipes: Optional[Dict[str, Dict[str, Any]]] = None,
        random_state: int = 42,
        stability_weight: float = 0.5,
    ):
        self.recipes = recipes or DEFAULT_RECIPES
        self.random_state = random_state
        self.stability_weight = stability_weight

    def list_recipes(self) -> List[str]:
        return list(self.recipes.keys())

    def _screen_models(self, task_type: str) -> List[Any]:
        if task_type == "classification":
            return [
                LogisticRegression(max_iter=500, random_state=self.random_state),
                RandomForestClassifier(
                    n_estimators=50, max_depth=8, random_state=self.random_state
                ),
            ]
        return [
            Ridge(random_state=self.random_state),
            RandomForestRegressor(
                n_estimators=50, max_depth=8, random_state=self.random_state
            ),
        ]

    def _score_fold(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        task_type: str,
        scoring: str,
    ) -> float:
        if task_type == "classification":
            return float(accuracy_score(y_true, y_pred))
        if scoring == "r2":
            return float(r2_score(y_true, y_pred))
        mse = float(np.mean((np.asarray(y_true) - y_pred) ** 2))
        return -mse

    def _composite_score(self, mean_score: float, std_score: float) -> float:
        return mean_score - self.stability_weight * std_score

    def _cv_recipe_score(
        self,
        recipe_name: str,
        recipe_cfg: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
        cv_folds: int,
        scoring: str,
    ) -> Tuple[float, float, float]:
        kf = KFold(
            n_splits=max(2, cv_folds),
            shuffle=True,
            random_state=self.random_state,
        )
        model_scores: List[float] = []
        for screen_model in self._screen_models(task_type):
            fold_scores: List[float] = []
            for train_idx, val_idx in kf.split(X):
                X_tr = X.iloc[train_idx]
                X_va = X.iloc[val_idx]
                y_tr = y.iloc[train_idx]
                y_va = y.iloc[val_idx]

                pipe = PreprocessingPipeline.from_recipe(recipe_name, recipe_cfg)
                X_tr_p, y_tr_p, _ = pipe.fit_transform(X_tr, y_tr, task_type=task_type)
                X_va_p = pipe.transform(X_va)

                model = screen_model
                model.fit(X_tr_p, y_tr_p)
                pred = model.predict(X_va_p)
                fold_scores.append(self._score_fold(y_va, pred, task_type, scoring))
            model_scores.append(float(np.mean(fold_scores)))

        mean_score = float(np.mean(model_scores))
        std_score = float(np.std(model_scores))
        composite = self._composite_score(mean_score, std_score)
        return composite, mean_score, std_score

    def search(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = "classification",
        cv_folds: int = 3,
        scoring: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Score each recipe with CV-safe preprocessing + dual screen models.

        Returns best recipe name, config, full-fit artifacts, and all scores.
        """
        scoring = scoring or default_scoring(task_type)
        results: List[Dict[str, Any]] = []
        best_name = "standard"
        best_composite = float("-inf")
        best_mean = 0.0
        best_std = 0.0

        for name, cfg in self.recipes.items():
            try:
                composite, mean_score, std_score = self._cv_recipe_score(
                    name, cfg, X, y, task_type, cv_folds, scoring
                )
                results.append(
                    {
                        "recipe": name,
                        "cv_mean": mean_score,
                        "cv_std": std_score,
                        "composite_score": composite,
                        "config": dict(cfg),
                    }
                )
                if composite > best_composite:
                    best_composite = composite
                    best_mean = mean_score
                    best_std = std_score
                    best_name = name
            except Exception as exc:
                logger.warning("Recipe %s failed: %s", name, exc)
                results.append({"recipe": name, "error": str(exc)})

        valid = [r for r in results if "composite_score" in r]
        valid.sort(key=lambda r: r["composite_score"], reverse=True)
        for row in valid:
            if row["recipe"] == best_name:
                row["selected"] = True
                if len(valid) > 1:
                    row["margin_over_runner_up"] = (
                        best_composite - valid[1]["composite_score"]
                    )
            else:
                row["delta_vs_winner"] = best_composite - row["composite_score"]

        best_cfg = dict(self.recipes.get(best_name, DEFAULT_RECIPES["standard"]))
        fit_pipe = PreprocessingPipeline.from_recipe(best_name, best_cfg)
        X_fit, y_fit, artifacts = fit_pipe.fit_transform(X, y, task_type=task_type)

        return {
            "preprocessing_recipe": best_name,
            "preprocessing_recipe_config": best_cfg,
            "preprocessing_search_results": results,
            "preprocessing_cv_mean": best_mean,
            "preprocessing_cv_std": best_std,
            "preprocessing_composite_score": best_composite,
            "preprocessing_scoring": scoring,
            "X_processed": X_fit,
            "y_processed": y_fit,
            "preprocessing_artifacts": artifacts,
        }
