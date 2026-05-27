"""
Integrated per-algorithm hyperparameter search during model selection.
"""

import itertools
import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from models.registry import MODEL_SEARCH_SPACE, DEFAULT_PARAMS
from optimizer.search_space import get_search_space

logger = logging.getLogger(__name__)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False

try:
    from core.adaptive_resource_manager import adaptive_resource_manager
except ImportError:
    adaptive_resource_manager = None

def _resolve_scoring(task_type: str, scoring: Optional[str] = None) -> str:
    if scoring:
        return scoring
    return "accuracy" if task_type == "classification" else "neg_mean_squared_error"


SEARCH_DEPTH_TRIALS = {
    "fast": 0,
    "balanced": 8,
    "deep": 50,
}


def get_search_budget(
    search_depth: str = "balanced",
    max_trials: Optional[int] = None,
    max_time: Optional[float] = None,
) -> Dict[str, Any]:
    """Resource-aware trial and timeout caps per model."""
    base_trials = max_trials or SEARCH_DEPTH_TRIALS.get(search_depth, 8)
    budget = {
        "max_trials_per_model": base_trials,
        "timeout_per_model": max_time or (120 if search_depth == "deep" else 45),
    }
    if adaptive_resource_manager is not None:
        adjusted = adaptive_resource_manager.adjust_parameters_for_resources(
            {"n_trials": base_trials, "cv_folds": 5}
        )
        budget["max_trials_per_model"] = adjusted.get("n_trials", base_trials)
    return budget


def _normalize_model_key(model_name: str) -> str:
    name = model_name.lower()
    for key in MODEL_SEARCH_SPACE:
        if key in name or name in key:
            return key
    return model_name.lower().split("_")[0]


def _split_space(space: Dict[str, Any]) -> tuple:
    discrete: Dict[str, Any] = {}
    continuous: Dict[str, tuple] = {}
    for key, spec in space.items():
        if isinstance(spec, list):
            discrete[key] = spec
        elif isinstance(spec, tuple) and len(spec) == 2:
            continuous[key] = spec
        else:
            discrete[key] = [spec]
    return discrete, continuous


def _expand_param_grid(space: Dict[str, Any], max_combinations: int) -> List[Dict[str, Any]]:
    discrete, continuous = _split_space(space)
    merged = {**discrete}
    for key, (lo, hi) in continuous.items():
        if isinstance(lo, int) and isinstance(hi, int):
            merged[key] = [lo, (lo + hi) // 2, hi][:3]
        else:
            merged[key] = [lo, (lo + hi) / 2, hi]

    keys = list(merged.keys())
    value_lists = [merged[k] if isinstance(merged[k], list) else [merged[k]] for k in keys]
    all_combos = [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]
    if len(all_combos) <= max_combinations:
        return all_combos
    random.shuffle(all_combos)
    return all_combos[:max_combinations]


def _optuna_search(
    model_factory: Callable,
    X: pd.DataFrame,
    y: pd.Series,
    space: Dict[str, Any],
    task_type: str,
    cv_folds: int,
    n_jobs: int,
    n_trials: int,
    timeout: Optional[float],
    random_state: int,
    scoring: str,
) -> Optional[Dict[str, Any]]:
    if not OPTUNA_AVAILABLE or n_trials <= 0:
        return None

    _, continuous = _split_space(space)
    if not continuous:
        return None

    def objective(trial):
        params = {}
        for key, spec in space.items():
            if isinstance(spec, list):
                params[key] = trial.suggest_categorical(key, spec)
            elif isinstance(spec, tuple) and len(spec) == 2:
                lo, hi = spec
                if isinstance(lo, int) and isinstance(hi, int):
                    params[key] = trial.suggest_int(key, lo, hi)
                else:
                    params[key] = trial.suggest_float(key, float(lo), float(hi))
        try:
            model = model_factory(**params)
            scores = cross_val_score(
                model, X, y, cv=cv_folds, scoring=scoring, n_jobs=n_jobs
            )
            return float(scores.mean())
        except Exception:
            return float("-inf")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    if study.best_trial.value is None or study.best_trial.value == float("-inf"):
        return None

    best_params = study.best_params
    model = model_factory(**best_params)
    model.fit(X, y)
    return {
        "model": model,
        "cv_mean": float(study.best_value),
        "cv_std": 0.0,
        "best_params": best_params,
        "trials_run": len(study.trials),
        "search_backend": "optuna",
    }


class ModelSearch:
    """Search hyperparameters per algorithm during model selection."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.search_history: List[Dict[str, Any]] = []

    def search_model(
        self,
        model_name: str,
        model_factory,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
        search_depth: str = "balanced",
        cv_folds: int = 5,
        n_jobs: int = -1,
        max_trials: Optional[int] = None,
        timeout_per_model: Optional[float] = None,
        evaluate_variants: bool = False,
        scoring: Optional[str] = None,
        best_cv_so_far: Optional[float] = None,
        early_stop_margin: float = 0.05,
    ) -> Dict[str, Any]:
        scoring = _resolve_scoring(task_type, scoring)

        start = time.time()

        if search_depth == "fast":
            result = self._cv_defaults(
                model_name, model_factory, X, y, task_type, scoring, cv_folds, n_jobs
            )
            result["train_time"] = time.time() - start
            return result

        if best_cv_so_far is not None and best_cv_so_far > float("-inf"):
            try:
                quick = cross_val_score(
                    model_factory(), X, y, cv=2, scoring=scoring, n_jobs=n_jobs
                )
                if float(quick[0]) < best_cv_so_far - early_stop_margin:
                    logger.info(
                        "%s skipped by fast screen (fold0=%.4f vs best=%.4f)",
                        model_name,
                        quick[0],
                        best_cv_so_far,
                    )
                    return {
                        "model": model_factory(),
                        "cv_scores": quick.tolist(),
                        "cv_mean": float(quick.mean()),
                        "cv_std": float(quick.std()),
                        "scoring": scoring,
                        "best_params": {},
                        "search_depth": search_depth,
                        "trials_run": 0,
                        "early_stopped": True,
                        "skip_reason": "Fast screen below leader margin",
                        "train_time": time.time() - start,
                    }
            except Exception as exc:
                logger.debug("Fast screen failed for %s: %s", model_name, exc)

        budget = get_search_budget(search_depth, max_trials, timeout_per_model)
        trials = budget["max_trials_per_model"]
        timeout = budget["timeout_per_model"]

        model_key = _normalize_model_key(model_name)
        space = get_search_space(model_key, task_type) or MODEL_SEARCH_SPACE.get(model_key, {})

        if evaluate_variants and search_depth in ("balanced", "deep"):
            from execution.preprocessing_variants import evaluate_preprocessing_variants
            variant_info = evaluate_preprocessing_variants(
                model_factory, X, y, task_type, cv_folds=min(3, cv_folds), n_jobs=n_jobs
            )
        else:
            variant_info = None

        if search_depth == "deep":
            param_grid = _expand_param_grid(space, max_combinations=9999)
        else:
            param_grid = _expand_param_grid(space, max_combinations=max(1, trials // 2))

        if not param_grid:
            defaults = DEFAULT_PARAMS.get(model_key, {})
            param_grid = [defaults] if defaults else [{}]

        best_score = float("-inf")
        best_model = None
        best_params: Dict[str, Any] = {}
        all_trials: List[Dict[str, Any]] = []

        for params in param_grid:
            if time.time() - start > timeout:
                break
            try:
                model = model_factory(**params)
                cv_scores = cross_val_score(
                    model, X, y, cv=cv_folds, scoring=scoring, n_jobs=n_jobs
                )
                mean_score = float(cv_scores.mean())
                trial = {
                    "params": params,
                    "cv_mean": mean_score,
                    "cv_std": float(cv_scores.std()),
                    "backend": "grid",
                }
                all_trials.append(trial)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                    best_model = model_factory(**params)
                    best_model.fit(X, y)
            except Exception as exc:
                logger.debug("Trial failed for %s: %s", model_name, exc)

        remaining = max(1, trials - len(all_trials))
        remaining_time = max(1.0, timeout - (time.time() - start))
        optuna_result = _optuna_search(
            model_factory,
            X,
            y,
            space,
            task_type,
            cv_folds,
            n_jobs,
            n_trials=remaining,
            timeout=remaining_time,
            random_state=self.random_state,
            scoring=scoring,
        )
        if optuna_result and optuna_result["cv_mean"] > best_score:
            best_score = optuna_result["cv_mean"]
            best_params = optuna_result["best_params"]
            best_model = optuna_result["model"]
            all_trials.append(
                {
                    "params": best_params,
                    "cv_mean": best_score,
                    "cv_std": optuna_result["cv_std"],
                    "backend": "optuna",
                }
            )

        if best_model is None:
            result = self._cv_defaults(
                model_name, model_factory, X, y, task_type, scoring, cv_folds, n_jobs
            )
            result["train_time"] = time.time() - start
            return result

        result = {
            "model": best_model,
            "cv_scores": [best_score],
            "cv_mean": best_score,
            "cv_std": all_trials[0]["cv_std"] if all_trials else 0.0,
            "scoring": scoring,
            "best_params": best_params,
            "search_depth": search_depth,
            "trials_run": len(all_trials),
            "all_trials": all_trials[:10],
            "preprocessing_variant": variant_info,
            "train_time": time.time() - start,
        }
        self.search_history.append({"model": model_name, **result})
        logger.info(
            "✅ %s [%s]: %.4f (params=%s, trials=%d)",
            model_name,
            search_depth,
            best_score,
            best_params,
            len(all_trials),
        )
        return result

    def _cv_defaults(
        self,
        model_name: str,
        model_factory,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
        scoring: str,
        cv_folds: int,
        n_jobs: int,
    ) -> Dict[str, Any]:
        model = model_factory()
        cv_scores = cross_val_score(
            model, X, y, cv=cv_folds, scoring=scoring, n_jobs=n_jobs
        )
        model.fit(X, y)
        return {
            "model": model,
            "cv_scores": cv_scores.tolist(),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "scoring": scoring,
            "best_params": {},
            "search_depth": "fast",
            "trials_run": 1,
        }
