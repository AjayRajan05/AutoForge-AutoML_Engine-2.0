"""Preprocessing recipe search tests."""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.unified_automl import UnifiedAutoML
from execution.preprocessing_search import PreprocessingSearchSpace, DEFAULT_RECIPES
from input_output.input_types import AutoMLInput


@pytest.fixture
def tabular_clf_df():
    X, y = make_classification(
        n_samples=80,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(X.shape[1])])
    df["cat"] = np.random.choice(["a", "b", "c"], size=len(df))
    df.loc[df.index[:5], "num_0"] = np.nan
    df["target"] = y
    return df


def test_search_space_has_four_recipes():
    space = PreprocessingSearchSpace()
    names = space.list_recipes()
    assert len(names) >= 4
    for key in ("standard", "robust", "minimal", "poly"):
        assert key in names
        assert key in DEFAULT_RECIPES


def test_search_returns_winner_and_results(tabular_clf_df):
    df = tabular_clf_df
    X = df.drop(columns=["target"])
    y = df["target"]
    searcher = PreprocessingSearchSpace(random_state=42)
    result = searcher.search(X, y, task_type="classification", cv_folds=3)
    assert result["preprocessing_recipe"] in searcher.list_recipes()
    assert len(result["preprocessing_search_results"]) >= 4
    assert result["X_processed"].shape[0] == len(X)
    assert "preprocessing_artifacts" in result


def test_balanced_fit_stores_preprocessing_recipe(tabular_clf_df):
    inp = AutoMLInput(
        data=tabular_clf_df,
        target_column="target",
        task_type="classification",
        search_depth="balanced",
        max_trials=2,
        random_state=42,
    )
    automl = UnifiedAutoML(
        {
            "search_depth": "balanced",
            "verbose": False,
            "enable_ensemble": False,
            "use_processors": False,
        }
    )
    automl.fit(inp, enable_optimization=False, enable_tracking=False, enable_monitoring=False)
    meta = automl.training_metadata
    assert meta.get("preprocessing_recipe") in PreprocessingSearchSpace().list_recipes()
    assert meta.get("preprocessing_search_results")


def test_fast_skips_preprocessing_search(tabular_clf_df):
    inp = AutoMLInput(
        data=tabular_clf_df,
        target_column="target",
        task_type="classification",
        search_depth="fast",
        random_state=42,
    )
    automl = UnifiedAutoML(
        {"search_depth": "fast", "verbose": False, "enable_ensemble": False, "use_processors": False}
    )
    automl.fit(inp, enable_optimization=False, enable_tracking=False, enable_monitoring=False)
    assert automl.training_metadata.get("preprocessing_search_results") is None


def test_regression_preprocessing_search():
    X, y = make_regression(n_samples=70, n_features=4, noise=0.3, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    searcher = PreprocessingSearchSpace(random_state=0)
    result = searcher.search(
        df.drop(columns=["target"]),
        df["target"],
        task_type="regression",
        cv_folds=3,
        scoring="neg_mean_squared_error",
    )
    assert result["preprocessing_recipe"] in searcher.list_recipes()
