"""Predict with unseen categorical values should not crash."""

import os
import sys

import pandas as pd
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.unified_automl import UnifiedAutoML
from input_output.input_types import AutoMLInput


@pytest.fixture
def cat_df():
    return pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
        "color": (list("redblue") * 5)[:20],
        "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


def test_predict_unseen_category(cat_df):
    inp = AutoMLInput(
        data=cat_df,
        target_column="y",
        task_type="classification",
        search_depth="fast",
    )
    automl = UnifiedAutoML({"search_depth": "fast", "verbose": False, "enable_ensemble": False})
    automl.fit(inp, enable_optimization=False, enable_tracking=False, enable_monitoring=False)

    unseen = pd.DataFrame({
        "num": [5.5],
        "color": ["purple"],
    })
    preds = automl.predict(unseen)
    assert len(preds) == 1
