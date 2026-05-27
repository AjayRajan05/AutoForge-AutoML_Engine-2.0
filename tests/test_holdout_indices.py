"""Verify holdout row indices never enter feature selection or preprocessing fit."""

import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.datasets import make_classification

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.unified_automl import UnifiedAutoML
from execution.preprocessing_pipeline import PreprocessingPipeline
from input_output.input_types import AutoMLInput


@pytest.fixture
def clf_df():
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=5,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df


def test_holdout_indices_excluded_from_feature_selection_and_preprocessing(clf_df):
    """Spy on fit inputs; holdout index set must not overlap train fit indices."""
    fit_indices_seen = []
    select_indices_seen = []
    original_fit_transform = PreprocessingPipeline.fit_transform

    def spy_select_features(X, y, profile=None, config=None):
        select_indices_seen.extend(list(X.index))
        return {
            "selected_features": list(X.columns),
            "feature_scores": {c: 1.0 for c in X.columns},
            "selection_ratio": 1.0,
            "feature_selection_on_train_only": True,
        }

    def spy_fit_transform(self, X, y, task_type="classification"):
        fit_indices_seen.append(set(X.index))
        return original_fit_transform(self, X, y, task_type=task_type)

    automl = UnifiedAutoML({
        "search_depth": "fast",
        "verbose": False,
        "enable_ensemble": False,
        "use_processors": False,
    })
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
    ), patch.object(
        PreprocessingPipeline,
        "fit_transform",
        spy_fit_transform,
    ):
        automl.fit(inp, enable_optimization=False, enable_tracking=False, enable_monitoring=False)

    holdout_idx = set(automl._holdout_split_["X_holdout"].index)
    train_idx = set(clf_df.index) - holdout_idx
    train_size = automl._holdout_split_["train_size"]

    assert select_indices_seen
    assert not holdout_idx.intersection(set(select_indices_seen))
    assert set(select_indices_seen).issubset(train_idx)
    assert len(select_indices_seen) == train_size

    for seen in fit_indices_seen:
        assert not holdout_idx.intersection(seen)
        assert seen.issubset(train_idx)
        assert len(seen) == train_size
