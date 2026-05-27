"""
Sklearn-compatible estimators for AutoForge.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from .unified_automl import UnifiedAutoML
from input_output.input_types import AutoMLInput


def _build_automl_input(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    target_column: str = "target",
    **kwargs,
) -> AutoMLInput:
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(np.asarray(X).shape[1])])
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name=target_column)
    else:
        y = y.rename(target_column)
    data = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    return AutoMLInput(data=data, target_column=target_column, **kwargs)


class _AutoForgeBase(BaseEstimator):
    """Shared sklearn interface delegating to UnifiedAutoML."""

    _task_type: Optional[str] = None

    def __init__(
        self,
        model_family: str = "ml",
        max_trials: Optional[int] = None,
        max_time: Optional[float] = None,
        user_preference: str = "auto",
        enable_optimization: bool = True,
        search_depth: str = "balanced",
        scoring: Optional[str] = None,
        target_column: str = "target",
        auto_report: bool = False,
        report_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.model_family = model_family
        self.max_trials = max_trials
        self.max_time = max_time
        self.user_preference = user_preference
        self.enable_optimization = enable_optimization
        self.search_depth = search_depth
        self.scoring = scoring
        self.target_column = target_column
        self.auto_report = auto_report
        self.report_dir = report_dir
        self.config = config or {}
        self._automl: Optional[UnifiedAutoML] = None

    def _make_config(self) -> Dict[str, Any]:
        cfg = dict(self.config)
        cfg.setdefault("model_family", self.model_family)
        cfg.setdefault("search_depth", self.search_depth)
        if self.auto_report:
            cfg.setdefault("auto_report", True)
        if self.report_dir:
            cfg.setdefault("report_dir", self.report_dir)
        return cfg

    def fit(self, X, y=None, target: Optional[str] = None):
        """Fit on (X, y) or on a DataFrame with ``target`` column name."""
        if y is None and isinstance(X, pd.DataFrame) and target:
            y = X[target]
            X = X.drop(columns=[target])
            self.target_column = target
        if y is None:
            raise ValueError("Provide y or fit(df, target='column_name')")
        automl_input = _build_automl_input(
            X,
            y,
            target_column=self.target_column,
            task_type=self._task_type,
            model_family=self.model_family,
            max_trials=self.max_trials,
            max_time=self.max_time,
            user_preference=self.user_preference,
            search_depth=self.search_depth,
            scoring=self.scoring,
        )
        self._automl = UnifiedAutoML(self._make_config())
        self._automl.fit(automl_input, enable_optimization=self.enable_optimization)
        return self

    def predict(self, X):
        self._check_fitted()
        return self._automl.predict(X)

    def predict_proba(self, X):
        self._check_fitted()
        return self._automl.predict_proba(X)

    def score(self, X, y):
        self._check_fitted()
        preds = self.predict(X)
        y_arr = np.asarray(y)
        task_type = getattr(self._automl, "task_type_", "classification")
        if task_type == "regression":
            from sklearn.metrics import r2_score
            return r2_score(y_arr, preds)
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_arr, preds)

    def get_model_comparison(self):
        self._check_fitted()
        return self._automl.get_model_comparison()

    def print_model_comparison(self):
        self._check_fitted()
        return self._automl.print_model_comparison()

    def get_selection_report(self):
        self._check_fitted()
        return self._automl.get_selection_report()

    def selection_summary(self):
        self._check_fitted()
        return self._automl.selection_summary()

    def get_params(self, deep=True):
        return {
            "model_family": self.model_family,
            "max_trials": self.max_trials,
            "max_time": self.max_time,
            "user_preference": self.user_preference,
            "enable_optimization": self.enable_optimization,
            "search_depth": self.search_depth,
            "scoring": self.scoring,
            "target_column": self.target_column,
            "auto_report": self.auto_report,
            "report_dir": self.report_dir,
            "config": self.config,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _check_fitted(self):
        if self._automl is None or not self._automl.is_fitted:
            raise ValueError("This AutoForge estimator is not fitted yet.")

    @property
    def automl_(self) -> UnifiedAutoML:
        self._check_fitted()
        return self._automl


class AutoForgeClassifier(_AutoForgeBase, ClassifierMixin):
    _task_type = "classification"


class AutoForgeRegressor(_AutoForgeBase, RegressorMixin):
    _task_type = "regression"


class AutoForgeEstimator(_AutoForgeBase):
    """Auto-detect classification vs regression from target values."""

    def fit(self, X, y=None, target: Optional[str] = None):
        if y is None and isinstance(X, pd.DataFrame) and target:
            y_series = X[target]
        elif y is not None:
            y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
        else:
            raise ValueError("Provide y or fit(df, target='column_name')")
        if len(y_series.unique()) < 20 and not pd.api.types.is_float_dtype(y_series):
            self._task_type = "classification"
        else:
            self._task_type = "regression"
        return super().fit(X, y, target=target)


def autoforge_fit(
    df: pd.DataFrame,
    target: str,
    task_type: Optional[str] = None,
    search_depth: str = "balanced",
    enable_optimization: bool = True,
    verbose: bool = True,
    **kwargs,
) -> UnifiedAutoML:
    """Three-line UX: train and return fitted UnifiedAutoML."""
    config = {"search_depth": search_depth, "verbose": verbose}
    config.update(kwargs.pop("config", {}) or {})
    automl = UnifiedAutoML(config)
    inp = AutoMLInput(
        data=df,
        target_column=target,
        task_type=task_type,
        search_depth=search_depth,
        **kwargs,
    )
    automl.fit(inp, enable_optimization=enable_optimization)
    return automl
