"""Predict edge-case validation."""

import os
import sys

import pandas as pd
import pytest
from sklearn.datasets import make_classification

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.unified_automl import UnifiedAutoML
from input_output.input_types import AutoMLInput


@pytest.fixture
def fitted_automl():
    X, y = make_classification(n_samples=60, n_features=4, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y
    inp = AutoMLInput(
        data=df, target_column="target", task_type="classification", search_depth="fast"
    )
    automl = UnifiedAutoML({"verbose": False, "enable_ensemble": False})
    automl.fit(inp, enable_optimization=False, enable_tracking=False, enable_monitoring=False)
    return automl, df


def test_predict_missing_column_raises(fitted_automl):
    automl, df = fitted_automl
    bad = df.drop(columns=["f0"])
    with pytest.raises(ValueError, match="Missing required feature columns"):
        automl.predict(bad)


def test_predict_extra_column_ok(fitted_automl):
    automl, df = fitted_automl
    X = df.drop(columns=["target"]).copy()
    X["extra"] = 1.0
    preds = automl.predict(X.head(5))
    assert len(preds) == 5


def test_predict_not_fitted_raises():
    automl = UnifiedAutoML()
    with pytest.raises(ValueError):
        automl.predict(pd.DataFrame({"a": [1, 2]}))
