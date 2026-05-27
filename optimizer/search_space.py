"""
Hyperparameter search space definitions for AutoForge optimizers.
"""

from typing import Any, Dict, List, Tuple, Union

try:
    from ..models.registry import MODEL_SEARCH_SPACE
except ImportError:
    MODEL_SEARCH_SPACE = {}


def get_search_space(model_name: str, task_type: str = "classification") -> Dict[str, Any]:
    """Return search space for a model type."""
    key = model_name.lower().replace("classifier", "").replace("regressor", "").strip("_")
    for name, space in MODEL_SEARCH_SPACE.items():
        if name in key or key in name:
            return space.copy()
    return SearchSpace().get_default_space(task_type)


class SearchSpace:
    """Defines hyperparameter search spaces for common models."""

    def get_random_forest_classification_space(self) -> Dict[str, Tuple]:
        return {
            "n_estimators": (50, 300, "int"),
            "max_depth": (3, 20, "int"),
            "min_samples_split": (2, 20, "int"),
            "min_samples_leaf": (1, 10, "int"),
        }

    def get_random_forest_regression_space(self) -> Dict[str, Tuple]:
        return self.get_random_forest_classification_space()

    def get_xgboost_classification_space(self) -> Dict[str, Tuple]:
        return {
            "n_estimators": (50, 300, "int"),
            "max_depth": (3, 12, "int"),
            "learning_rate": (0.01, 0.3, "float"),
            "subsample": (0.6, 1.0, "float"),
        }

    def get_xgboost_regression_space(self) -> Dict[str, Tuple]:
        return self.get_xgboost_classification_space()

    def get_default_space(self, task_type: str = "classification") -> Dict[str, Any]:
        return {
            "random_state": [42],
            "n_jobs": [-1],
        }

    def for_model(self, model_type: str, task_type: str) -> Dict[str, Any]:
        model_type = model_type.lower()
        if "randomforest" in model_type or "random_forest" in model_type:
            if task_type == "classification":
                return self.get_random_forest_classification_space()
            return self.get_random_forest_regression_space()
        if "xgb" in model_type:
            if task_type == "classification":
                return self.get_xgboost_classification_space()
            return self.get_xgboost_regression_space()
        return get_search_space(model_type, task_type)
