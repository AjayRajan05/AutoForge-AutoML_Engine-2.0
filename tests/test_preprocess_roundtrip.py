"""Train/predict round-trip using unified preprocessing pipeline."""

import pandas as pd
import pytest

from core.unified_automl import UnifiedAutoML
from input_output.input_types import AutoMLInput


@pytest.fixture
def mixed_tabular_df():
    return pd.DataFrame({
        "num_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                  11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
        "num_b": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                  1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        "cat": list("abababababababababab"),
        "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


def test_preprocess_train_predict_feature_count(mixed_tabular_df):
    inp = AutoMLInput(
        data=mixed_tabular_df,
        target_column="y",
        task_type="classification",
        search_depth="fast",
    )
    automl = UnifiedAutoML({"search_depth": "fast", "verbose": False})
    automl.fit(inp, enable_optimization=False, enable_tracking=False, enable_monitoring=False)
    X = mixed_tabular_df.drop(columns=["y"])
    preds = automl.predict(X)
    assert len(preds) == len(X)
    train_n = len(automl.training_metadata.get("feature_names", []))
    if train_n:
        processed = automl._apply_preprocessing(X.head(1))
        proc_cols = processed.shape[1] if hasattr(processed, "shape") else len(processed[0])
        assert proc_cols == train_n
