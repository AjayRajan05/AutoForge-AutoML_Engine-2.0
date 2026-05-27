"""Shared pytest configuration."""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


@pytest.fixture(scope="session")
def fast_automl_config():
    """Minimal config for fast integration tests."""
    return {
        "search_depth": "fast",
        "verbose": False,
        "enable_ensemble": False,
    }

