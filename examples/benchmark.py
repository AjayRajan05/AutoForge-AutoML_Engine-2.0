#!/usr/bin/env python
"""Run classical benchmark suite and print report path."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarking.classical_suite import (  # noqa: E402
    evaluate_exit_criteria,
    run_classical_suite,
    write_benchmarks_md,
)


def main():
    datasets = sys.argv[1:] if len(sys.argv) > 1 else None
    results = run_classical_suite(
        datasets=datasets,
        depths=["fast", "balanced"],
        test_roundtrip=True,
    )
    path = write_benchmarks_md(results)
    print(f"Report: {path.resolve()}")
    print("Criteria:", evaluate_exit_criteria(results))


if __name__ == "__main__":
    main()
