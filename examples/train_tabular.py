#!/usr/bin/env python
"""Minimal tabular training example."""

import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autoforge import AutoForgeClassifier  # noqa: E402


def main():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=[f"f{i}" for i in range(iris.data.shape[1])])
    df["target"] = iris.target

    model = AutoForgeClassifier(search_depth="fast")
    model.fit(df, target="target")
    print(model.selection_summary())
    print(f"Score on sample: {model.score(df.drop(columns=['target']).head(20), df['target'].head(20)):.4f}")


if __name__ == "__main__":
    main()
