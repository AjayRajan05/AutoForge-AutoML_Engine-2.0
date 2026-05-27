"""Save/load round-trip and predict parity tests."""

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.unified_automl import UnifiedAutoML
from input_output.input_types import AutoMLInput


@pytest.fixture
def clf_df():
    X, y = make_classification(n_samples=80, n_features=6, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = y
    return df


@pytest.fixture
def reg_df():
    X, y = make_regression(n_samples=80, n_features=5, noise=0.5, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    return df


def _fit(df, task_type, depth="fast"):
    inp = AutoMLInput(
        data=df,
        target_column="target",
        task_type=task_type,
        search_depth=depth,
        random_state=42,
    )
    automl = UnifiedAutoML({
        "search_depth": depth,
        "verbose": False,
        "enable_ensemble": False,
        "use_processors": False,
    })
    automl.fit(inp, enable_optimization=False, enable_tracking=False, enable_monitoring=False)
    return automl


@pytest.mark.parametrize("task_type,df_fixture", [
    ("classification", "clf_df"),
    ("regression", "reg_df"),
])
def test_save_load_predict_roundtrip(task_type, df_fixture, request):
    df = request.getfixturevalue(df_fixture)
    automl = _fit(df, task_type)
    holdout = automl._holdout_split_["X_holdout"]
    expected = automl.predict(holdout.head(5))

    name = f"_rt_{task_type}"
    try:
        automl.save_model(name, overwrite=True)
        loaded = UnifiedAutoML({"verbose": False})
        loaded.load_model(name)
        preds = loaded.predict(holdout.head(5))
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(preds))
    finally:
        shutil.rmtree(Path("models") / name, ignore_errors=True)


def test_balanced_preprocessing_recipe_roundtrip(clf_df):
    automl = _fit(clf_df, "classification", depth="balanced")
    assert automl.training_metadata.get("preprocessing_recipe") is not None
    holdout = automl._holdout_split_["X_holdout"]
    preds = automl.predict(holdout.head(3))
    assert len(preds) == 3
