"""Balanced HPO integration test (slow)."""

import numpy as np
import pandas as pd
import pytest

from core.unified_automl import UnifiedAutoML
from input_output.input_types import AutoMLInput


@pytest.fixture
def tiny_regression_data():
    rng = np.random.RandomState(0)
    n = 60
    df = pd.DataFrame({
        'x1': rng.randn(n),
        'x2': rng.randn(n),
        'cat': rng.choice(['a', 'b'], n),
        'y': rng.randn(n) * 2 + 1,
    })
    return df


@pytest.mark.slow
def test_balanced_search_depth_completes(tiny_regression_data):
    df = tiny_regression_data
    inp = AutoMLInput(
        data=df,
        target_column='y',
        task_type='regression',
        search_depth='balanced',
        max_trials=3,
        user_preference='fast',
    )
    automl = UnifiedAutoML({
        'search_depth': 'balanced',
        'verbose': False,
        'use_processors': True,
    })
    automl.fit(inp, enable_optimization=True)
    assert automl.is_fitted
    assert getattr(automl, 'search_depth_', 'balanced') == 'balanced'
    comparison = automl.get_model_comparison()
    assert comparison is not None


def test_fast_search_skips_refinement_by_default(tiny_regression_data):
    df = tiny_regression_data
    inp = AutoMLInput(
        data=df,
        target_column='y',
        task_type='regression',
        search_depth='fast',
        user_preference='fast',
    )
    automl = UnifiedAutoML({'search_depth': 'fast', 'verbose': False})
    automl.fit(inp, enable_optimization=False)
    assert automl.is_fitted
