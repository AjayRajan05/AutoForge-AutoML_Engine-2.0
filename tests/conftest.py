"""Shared pytest configuration."""

import pytest


@pytest.fixture(scope="session")
def fast_automl_config():
    """Minimal config for fast integration tests."""
    return {
        "search_depth": "fast",
        "verbose": False,
        "enable_ensemble": False,
    }

