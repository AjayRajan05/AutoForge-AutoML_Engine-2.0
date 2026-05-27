"""Tests for integrated model search."""

import numpy as np
import pandas as pd
import pytest

from execution.model_search import ModelSearch, SEARCH_DEPTH_TRIALS


@pytest.fixture
def small_classification_data():
    rng = np.random.RandomState(42)
    X = pd.DataFrame({
        'a': rng.randn(80),
        'b': rng.randn(80),
        'c': rng.randn(80),
    })
    y = pd.Series(rng.randint(0, 2, 80))
    return X, y


def test_search_depth_trials_defined():
    assert SEARCH_DEPTH_TRIALS['fast'] == 0
    assert SEARCH_DEPTH_TRIALS['balanced'] > 0


def test_fast_mode_uses_defaults(small_classification_data):
    from registry.model_registry import model_registry

    X, y = small_classification_data
    search = ModelSearch(random_state=42)

    def factory(**params):
        return model_registry.get_model('random_forest', 'classification', **params)

    result = search.search_model(
        'random_forest', factory, X, y, 'classification',
        search_depth='fast', cv_folds=3, n_jobs=1,
    )
    assert 'cv_mean' in result
    assert result.get('search_depth') == 'fast'


def test_balanced_mode_runs_trials(small_classification_data):
    from registry.model_registry import model_registry

    X, y = small_classification_data
    search = ModelSearch(random_state=42)

    def factory(**params):
        return model_registry.get_model('random_forest', 'classification', **params)

    result = search.search_model(
        'random_forest', factory, X, y, 'classification',
        search_depth='balanced', cv_folds=3, n_jobs=1, max_trials=3,
    )
    assert result.get('trials_run', 0) >= 1
    assert 'best_params' in result
