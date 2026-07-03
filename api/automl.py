"""
Deprecated legacy AutoML wrapper — delegates to sklearn-style estimators.

Prefer:
    from autoforge import AutoForgeRegressor, AutoForgeClassifier
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from core.estimator import AutoForgeClassifier, AutoForgeRegressor

warnings.warn(
    "api.automl.AutoML is deprecated; use autoforge.AutoForgeRegressor or AutoForgeClassifier.",
    DeprecationWarning,
    stacklevel=2,
)


class AutoML:
    """Thin backward-compatible wrapper around AutoForge estimators."""

    def __init__(
        self,
        max_trials: Optional[int] = 10,
        random_state: int = 42,
        search_depth: str = "fast",
        **kwargs: Any,
    ):
        self.max_trials = max_trials
        self.random_state = random_state
        self.search_depth = search_depth
        self._kwargs = kwargs
        self._estimator: Optional[Union[AutoForgeClassifier, AutoForgeRegressor]] = None
        self._task_type: Optional[str] = None

    def _pick_estimator(self, y) -> Union[AutoForgeClassifier, AutoForgeRegressor]:
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
        if len(y_series.unique()) < 20 and not pd.api.types.is_float_dtype(y_series):
            self._task_type = "classification"
            return AutoForgeClassifier(
                max_trials=self.max_trials,
                search_depth=self.search_depth,
                config={"random_state": self.random_state, **self._kwargs},
            )
        self._task_type = "regression"
        return AutoForgeRegressor(
            max_trials=self.max_trials,
            search_depth=self.search_depth,
            config={"random_state": self.random_state, **self._kwargs},
        )

    def fit(self, X, y) -> "AutoML":
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        self._estimator = self._pick_estimator(y)
        self._estimator.fit(X, y)
        return self

    def predict(self, X):
        if self._estimator is None:
            raise ValueError("Model must be fitted before predict")
        return self._estimator.predict(X)

    @property
    def best_score(self) -> Optional[float]:
        if self._estimator and self._estimator.automl_:
            return self._estimator.automl_.best_score
        return None

    @property
    def best_model(self):
        if self._estimator and self._estimator.automl_:
            return self._estimator.automl_.best_model
        return None
