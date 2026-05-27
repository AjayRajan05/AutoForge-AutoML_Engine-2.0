"""
Quick smoke tests for AutoForge core workflow and integrations.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def test_basic_functionality():
    """Train, predict, save, and reload through UnifiedAutoML."""
    from core.unified_automl import UnifiedAutoML
    from input_output.input_types import AutoMLInput

    X, y = make_classification(
        n_samples=50,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_df = pd.Series(y, name="target")

    automl_input = AutoMLInput(
        data=pd.concat([X_df, y_df], axis=1),
        target_column="target",
        max_trials=3,
        max_time=15,
    )

    autoforge = UnifiedAutoML()
    autoforge.fit(
        automl_input,
        enable_tracking=False,
        enable_monitoring=False,
        enable_optimization=False,
    )

    predictions = autoforge.predict(X_df)
    assert autoforge.is_fitted
    assert len(predictions) == len(X_df)

    model_path = autoforge.save_model("quick_test_model")
    assert model_path

    new_autoforge = UnifiedAutoML()
    new_autoforge.load_model("quick_test_model")
    assert new_autoforge.is_fitted

    summary = autoforge.get_integration_summary()
    assert isinstance(summary, dict)
    assert len(summary) > 0


def test_integrations():
    """Import key integration modules without side effects."""
    from registry.feature_registry import feature_registry
    from registry.model_registry import model_registry
    from processors.processor_integration import processor_integrator
    from ensemble.ensemble_integration import ensemble_integrator
    from api.api_integration import api_integrator
    from benchmarking.benchmark_integration import benchmarking_integrator
    from systemization.systemization_integration import systemization_integrator
    from tracking.tracking_integration import tracking_integrator
    from optimizer.optimizer_integration import optimizer_integrator

    assert feature_registry is not None
    assert model_registry is not None
    assert processor_integrator is not None
    assert ensemble_integrator is not None
    assert api_integrator is not None
    assert benchmarking_integrator is not None
    assert systemization_integrator is not None
    assert tracking_integrator is not None
    assert optimizer_integrator is not None


def test_processor_tabular_path():
    """TabularProcessor is wired through ProcessorIntegrator."""
    from processors.processor_integration import processor_integrator

    X = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": ["x", "y", "x", "z"]})
    y = pd.Series([0, 1, 0, 1])
    X_out, y_out = processor_integrator.process_data(X, y, data_type="tabular")
    assert X_out.shape[0] == X.shape[0]
    assert len(y_out) == len(y)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
