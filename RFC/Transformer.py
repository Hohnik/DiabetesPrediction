import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ReplaceZeroWithMean(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns  # List of columns to transform

    def fit(self, X, y=None):
        # Compute the mean of non-zero values for each specified column
        self.means = {
            col: (
                X.loc[X[col] != 0, col].mean()  # For DataFrame
                if isinstance(X, pd.DataFrame)
                else np.nanmean(
                    np.where(X[:, col] != 0, X[:, col], np.nan)
                )  # For NumPy array
            )
            for col in self.columns
        }
        return self

    def transform(self, X, y=None):
        X = X.copy()  # Make a copy to avoid modifying the original data

        for col in self.columns:
            if isinstance(X, pd.DataFrame):
                # Ensure the mean value matches the column type to avoid dtype incompatibility
                X.loc[X[col] == 0, col] = X[col].dtype.type(self.means[col])
            else:
                # For NumPy arrays, use the dtype of the column (assumes the dtype is uniform for each column)
                X[X[:, col] == 0, col] = self.means[col]

        return X
