"""
Optimizer Integration — connects AutoForge with hyperparameter optimization modules.
"""

import logging
import time
from copy import deepcopy
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

try:
    from .adaptive_optimizer import AdaptiveOptimizer
except ImportError:
    AdaptiveOptimizer = None

try:
    from .optuna_search import OptunaOptimizer
except ImportError:
    OptunaOptimizer = None

try:
    from .search_space import SearchSpace, get_search_space
except ImportError:
    SearchSpace = None
    get_search_space = None

logger = logging.getLogger(__name__)


class OptimizerIntegrator:
    """Integration layer for optimization systems."""

    def __init__(self):
        self.available_optimizers = self._check_available_optimizers()
        self.optimization_history = []

    def _check_available_optimizers(self) -> Dict[str, bool]:
        optimizers = {
            "adaptive_optimizer": AdaptiveOptimizer is not None,
            "optuna_search": OptunaOptimizer is not None,
            "search_space": SearchSpace is not None,
        }
        logger.info(
            "Available optimizers: %s/%s",
            sum(optimizers.values()),
            len(optimizers),
        )
        return optimizers

    def create_search_space(self, model_type: str, task_type: str) -> Dict[str, Any]:
        try:
            if self.available_optimizers.get("search_space", False):
                return SearchSpace().for_model(model_type, task_type)
            return self._get_fallback_search_space(model_type, task_type)
        except Exception as exc:
            logger.warning("Search space creation failed: %s", exc)
            return self._get_fallback_search_space(model_type, task_type)

    def _get_fallback_search_space(self, model_type: str, task_type: str) -> Dict[str, Any]:
        if get_search_space is not None:
            return get_search_space(model_type, task_type)
        return {"random_state": [42], "n_jobs": [-1]}

    def _scoring(self, y: pd.Series) -> str:
        return "accuracy" if len(np.unique(y)) < 20 else "r2"

    def _evaluate(self, model: Any, X: pd.DataFrame, y: pd.Series, scoring: str) -> float:
        scores = cross_val_score(model, X, y, cv=3, scoring=scoring, n_jobs=-1)
        return float(scores.mean())

    def optimize_with_adaptive(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        search_space: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            if not self.available_optimizers.get("adaptive_optimizer", False):
                return self._fallback_optimization(model, X, y, search_space, config)

            scoring = self._scoring(y)
            model_template = deepcopy(model)
            max_trials = config.get("max_trials", 20)
            timeout = config.get("timeout", 300)
            start_time = time.time()

            def objective(params: Dict[str, Any]) -> float:
                trial_model = deepcopy(model_template)
                trial_model.set_params(**params)
                return self._evaluate(trial_model, X, y, scoring)

            optimizer = AdaptiveOptimizer(
                initial_trials=min(10, max_trials),
                max_trials=max_trials,
                time_budget=timeout,
            )

            if hasattr(optimizer, "optimize_with_search_space"):
                results = optimizer.optimize_with_search_space(
                    model_template, X, y, search_space, scoring=scoring
                )
            else:
                results = self._fallback_optimization(
                    model, X, y, search_space, config
                )
                return results

            best_params = results.get("best_params", {})
            best_score = results.get("best_score", -np.inf)
            if best_params:
                model.set_params(**best_params)

            formatted = {
                "best_params": best_params,
                "best_score": best_score,
                "best_model": model,
                "optimization_time": time.time() - start_time,
                "trials_completed": results.get("trials_completed", max_trials),
                "optimizer_type": "adaptive",
            }
            self.optimization_history.append(formatted)
            return formatted
        except Exception as exc:
            logger.error("Adaptive optimization failed: %s", exc)
            return self._fallback_optimization(model, X, y, search_space, config)

    def optimize_with_optuna(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        search_space: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            if not self.available_optimizers.get("optuna_search", False):
                return self._fallback_optimization(model, X, y, search_space, config)

            task_type = "classification" if self._scoring(y) == "accuracy" else "regression"
            start_time = time.time()
            optimizer = OptunaOptimizer(
                n_trials=config.get("max_trials", 20),
                cv=3,
                task_type=task_type,
            )
            study_result = optimizer.optimize(X, y)
            best_score = study_result.get("best_score", -np.inf) if isinstance(study_result, dict) else -np.inf
            best_params = study_result.get("best_params", {}) if isinstance(study_result, dict) else {}

            if best_params:
                model.set_params(**best_params)

            formatted = {
                "best_params": best_params,
                "best_score": best_score,
                "best_model": model,
                "optimization_time": time.time() - start_time,
                "trials_completed": config.get("max_trials", 20),
                "optimizer_type": "optuna",
            }
            self.optimization_history.append(formatted)
            return formatted
        except Exception as exc:
            logger.error("Optuna optimization failed: %s", exc)
            return self._fallback_optimization(model, X, y, search_space, config)

    def _fallback_optimization(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        search_space: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        scoring = self._scoring(y)
        best_score = -np.inf
        best_params: Dict[str, Any] = {}
        max_trials = config.get("max_trials", 20)

        for _ in range(max_trials):
            params = self._sample_parameters(search_space)
            try:
                trial_model = deepcopy(model)
                trial_model.set_params(**params)
                score = self._evaluate(trial_model, X, y, scoring)
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            except Exception as exc:
                logger.debug("Trial failed: %s", exc)

        if best_params:
            model.set_params(**best_params)

        results = {
            "best_params": best_params,
            "best_score": best_score,
            "best_model": model,
            "optimization_time": 0,
            "trials_completed": max_trials,
            "optimizer_type": "fallback",
        }
        self.optimization_history.append(results)
        return results

    def _sample_parameters(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        params = {}
        for param_name, param_config in search_space.items():
            if isinstance(param_config, list):
                params[param_name] = np.random.choice(param_config)
            elif isinstance(param_config, tuple) and len(param_config) >= 3:
                min_val, max_val, param_type = param_config[:3]
                if param_type == "int":
                    params[param_name] = int(np.random.randint(min_val, max_val + 1))
                else:
                    params[param_name] = float(np.random.uniform(min_val, max_val))
            else:
                params[param_name] = param_config
        return params

    def compare_optimizers(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        search_space: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        results = {}
        if self.available_optimizers.get("adaptive_optimizer", False):
            results["adaptive"] = self.optimize_with_adaptive(
                deepcopy(model), X, y, search_space, config
            )
        if self.available_optimizers.get("optuna_search", False):
            results["optuna"] = self.optimize_with_optuna(
                deepcopy(model), X, y, search_space, config
            )
        results["fallback"] = self._fallback_optimization(
            deepcopy(model), X, y, search_space, config
        )

        best_optimizer = None
        best_score = -np.inf
        for name, result in results.items():
            score = result.get("best_score", -np.inf)
            if score > best_score:
                best_score = score
                best_optimizer = name

        results["comparison"] = {
            "best_optimizer": best_optimizer,
            "best_score": best_score,
            "optimizers_tested": list(results.keys()),
        }
        return results


optimizer_integrator = OptimizerIntegrator()


def optimize_autoforge_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    optimizer_type: str = "adaptive",
    **kwargs,
) -> Dict[str, Any]:
    task_type = "classification" if len(np.unique(y)) < 20 else "regression"
    search_space = optimizer_integrator.create_search_space(type(model).__name__, task_type)

    if optimizer_type == "optuna":
        return optimizer_integrator.optimize_with_optuna(model, X, y, search_space, kwargs)
    if optimizer_type == "adaptive":
        return optimizer_integrator.optimize_with_adaptive(model, X, y, search_space, kwargs)
    return optimizer_integrator._fallback_optimization(model, X, y, search_space, kwargs)


def compare_autoforge_optimizers(model: Any, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
    task_type = "classification" if len(np.unique(y)) < 20 else "regression"
    search_space = optimizer_integrator.create_search_space(type(model).__name__, task_type)
    return optimizer_integrator.compare_optimizers(model, X, y, search_space, kwargs)
