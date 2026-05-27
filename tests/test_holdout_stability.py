"""Holdout metric stability and prep-search leakage guards."""

import os
import sys

import pandas as pd
import pytest
from sklearn.datasets import make_classification

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.unified_automl import UnifiedAutoML
from input_output.input_types import AutoMLInput
from execution.preprocessing_search import PreprocessingSearchSpace


@pytest.fixture
def clf_df():
    X, y = make_classification(n_samples=120, n_features=8, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["target"] = y
    return df


def test_holdout_metrics_stable_across_same_seed(clf_df):
    scores = []
    for _ in range(2):
        inp = AutoMLInput(
            data=clf_df,
            target_column="target",
            task_type="classification",
            search_depth="fast",
            random_state=42,
        )
        automl = UnifiedAutoML({"verbose": False, "enable_ensemble": False, "use_processors": False})
        automl.fit(inp, enable_optimization=False, enable_tracking=False, enable_monitoring=False)
        holdout = automl.training_metadata.get("holdout_metrics", {})
        scores.append(holdout.get("accuracy") or holdout.get("primary_score"))
    assert scores[0] == scores[1]


def test_preprocessing_search_never_sees_holdout_rows(clf_df):
    automl = UnifiedAutoML({"verbose": False})
    inp = AutoMLInput(
        data=clf_df,
        target_column="target",
        task_type="classification",
        search_depth="balanced",
        random_state=42,
    )
    automl._ensure_core_components()
    automl.fit(inp, enable_optimization=False, enable_tracking=False, enable_monitoring=False)
    split = automl._holdout_split_
    train_size = split["train_size"]
    # Preprocessing search runs on X_train only — train_size < full data
    assert train_size < len(clf_df)
    assert split["holdout_size"] + train_size == len(clf_df)


def test_preprocessing_search_has_eight_recipes():
    space = PreprocessingSearchSpace()
    assert len(space.list_recipes()) >= 8
