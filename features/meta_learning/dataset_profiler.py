import numpy as np
import pandas as pd


def profile_dataset(X, y):
    profile = {}

    if isinstance(X, pd.DataFrame):
        profile["num_rows"] = len(X)
        profile["num_cols"] = X.shape[1]
        profile["num_numeric"] = X.select_dtypes(include=np.number).shape[1]
        profile["num_categorical"] = X.select_dtypes(exclude=np.number).shape[1]
        profile["missing_ratio"] = X.isnull().sum().sum() / (X.size + 1e-9)
    else:
        profile["num_rows"] = X.shape[0]
        profile["num_cols"] = X.shape[1]
        profile["num_numeric"] = X.shape[1]
        profile["num_categorical"] = 0
        profile["missing_ratio"] = np.isnan(X).sum() / (X.size + 1e-9)

    profile["target_classes"] = len(set(y))

    return profile