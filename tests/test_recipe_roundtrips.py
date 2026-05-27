"""Parametrized fit/transform round-trip for all preprocessing recipes."""

import numpy as np
import pandas as pd
import pytest

from execution.preprocessing_pipeline import PreprocessingPipeline
from execution.preprocessing_search import DEFAULT_RECIPES


@pytest.fixture
def mixed_df():
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "num_a": rng.randn(40),
        "num_b": rng.randn(40),
        "cat": rng.choice(list("ABCD"), 40),
    })


@pytest.mark.parametrize("recipe_name", list(DEFAULT_RECIPES.keys()))
def test_recipe_fit_transform_roundtrip(mixed_df, recipe_name):
    y = pd.Series(np.random.RandomState(1).randint(0, 2, len(mixed_df)))
    pipe = PreprocessingPipeline.from_recipe(recipe_name, DEFAULT_RECIPES[recipe_name])
    X_train, _, artifacts = pipe.fit_transform(mixed_df, y, task_type="classification")
    assert X_train.shape[0] == len(mixed_df)

    replay = PreprocessingPipeline()
    replay.artifacts = artifacts
    replay.is_fitted = True
    replay.column_transformer = artifacts.get("column_transformer")
    X_test = replay.transform(mixed_df.iloc[:5])
    assert X_test.shape[0] == 5
    assert X_test.shape[1] == X_train.shape[1]
