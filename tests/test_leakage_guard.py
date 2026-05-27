"""Verify feature selection runs on train split only (leakage guard)."""

import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.datasets import make_classification

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.unified_automl import UnifiedAutoML
from input_output.input_types import AutoMLInput


@pytest.fixture
def clf_df():
    X, y = make_classification(
        n_samples=100,
        n_features=8,
        n_informative=5,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df


def test_feature_selection_uses_train_rows_only(clf_df):
    """FeatureSelector.select_features must receive ~80% of rows, not full dataset."""
    captured = {}

    def spy_select_features(X, y, profile=None, config=None):
        captured["n_rows"] = len(X)
        captured["n_cols"] = X.shape[1]
        return {
            "selected_features": list(X.columns),
            "feature_scores": {c: 1.0 for c in X.columns},
            "selection_ratio": 1.0,
        }

    automl = UnifiedAutoML({"search_depth": "fast", "verbose": False, "enable_ensemble": False})
    automl._ensure_core_components()

    inp = AutoMLInput(
        data=clf_df,
        target_column="target",
        task_type="classification",
        search_depth="fast",
        random_state=42,
    )

    with patch.object(
        automl.feature_selector,
        "select_features",
        side_effect=spy_select_features,
    ):
        automl.fit(inp, enable_optimization=False, enable_tracking=False, enable_monitoring=False)

    assert captured["n_rows"] == automl._holdout_split_["train_size"]
    assert captured["n_rows"] < len(clf_df)
    assert automl.training_metadata.get("feature_selection_on_train_only") is True


def test_holdout_not_used_in_feature_selection(clf_df):
    """Holdout size + train size equals full dataset after split."""
    automl = UnifiedAutoML({"search_depth": "fast", "verbose": False, "enable_ensemble": False})
    inp = AutoMLInput(
        data=clf_df,
        target_column="target",
        task_type="classification",
        search_depth="fast",
        random_state=42,
    )
    automl.fit(inp, enable_optimization=False, enable_tracking=False, enable_monitoring=False)
    split = automl._holdout_split_
    assert split["train_size"] + split["holdout_size"] == len(clf_df)
    assert split["holdout_size"] == pytest.approx(len(clf_df) * 0.2, abs=2)
