#!/usr/bin/env python
"""Load a saved model and predict."""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.unified_automl import UnifiedAutoML  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Saved model name under models/")
    parser.add_argument("--data", required=True, help="CSV without target column")
    parser.add_argument("--output", help="Optional output CSV path")
    args = parser.parse_args()

    automl = UnifiedAutoML({"verbose": False})
    automl.load_model(args.model)
    df = pd.read_csv(args.data)
    preds = automl.predict(df)
    out = df.copy()
    out["prediction"] = preds
    if args.output:
        out.to_csv(args.output, index=False)
        print(f"Wrote {args.output}")
    else:
        print(out.head())


if __name__ == "__main__":
    main()
