"""Smoke tests for classical benchmark suite."""

import json
import os
import sys
from pathlib import Path

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from benchmarking.classical_suite import (
    build_standard_datasets,
    run_classical_suite,
    write_benchmarks_md,
)


def test_build_standard_datasets():
    datasets = build_standard_datasets()
    assert "iris" in datasets
    assert "regression_synthetic" in datasets
    assert len(datasets["iris"]["df"]) == 150


@pytest.mark.slow
def test_classical_suite_iris_fast_only(tmp_path):
    results = run_classical_suite(
        datasets=["iris"],
        depths=["fast"],
        include_baseline=True,
    )
    assert len(results) >= 2
    assert any(r.system == "sklearn_baseline" and r.success for r in results)
    assert any(r.system == "autoforge_fast" and r.success for r in results)
    md = write_benchmarks_md(results, output_dir=str(tmp_path))
    assert md.exists()
    data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert isinstance(data, list) and len(data) >= 2
