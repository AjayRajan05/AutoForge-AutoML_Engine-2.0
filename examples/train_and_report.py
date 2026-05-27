#!/usr/bin/env python
"""
Train a tabular model with AutoForge and save the justification report bundle.

Usage:
    python examples/train_and_report.py --data data.csv --target price
    python examples/train_and_report.py  # uses synthetic regression data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_regression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.estimator import AutoForgeRegressor  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="AutoForge train + report example")
    parser.add_argument("--data", help="CSV path (optional)")
    parser.add_argument("--target", default="target", help="Target column name")
    parser.add_argument("--search-depth", default="balanced", choices=["fast", "balanced", "deep"])
    parser.add_argument("--save-model", default="example_model", help="Model name under models/")
    args = parser.parse_args()

    if args.data:
        df = pd.read_csv(args.data)
        target = args.target
    else:
        X, y = make_regression(n_samples=200, n_features=8, noise=0.5, random_state=42)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df["target"] = y
        target = "target"
        print("No --data provided; using synthetic regression dataset.")

    model = AutoForgeRegressor(search_depth=args.search_depth)
    model.fit(df, target=target)

    print(model.selection_summary())
    model_path = model.automl_.save_model(args.save_model)
    report_dir = Path("models") / args.save_model
    report_md = report_dir / "REPORT.md"

    print(f"\nModel saved: {model_path}")
    print(f"Report:      {report_md.resolve()}")
    if report_md.exists():
        print("\n--- REPORT.md (first 40 lines) ---")
        lines = report_md.read_text(encoding="utf-8").splitlines()
        for line in lines[:40]:
            print(line)


if __name__ == "__main__":
    main()
