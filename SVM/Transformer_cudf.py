<<<<<<< HEAD
import cudf
import cupy as cp
from sklearn.base import BaseEstimator, TransformerMixin


class ReplaceZeroWithMean(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns  # List of columns to transform

    def fit(self, X, y=None):
        # Compute the mean of non-zero values for each specified column
        self.means = {
            col: (
                X.loc[X[col] != 0, col].mean()  # For cuDF DataFrame
                if isinstance(X, cudf.DataFrame)
                else cp.nanmean(
                    cp.where(X[:, col] != 0, X[:, col], cp.nan)
                )  # For CuPy array
            )
            for col in self.columns
        }
        return self

    def transform(self, X, y=None):
        X = X.copy()  # Make a copy to avoid modifying the original data

        for col in self.columns:
            if isinstance(X, cudf.DataFrame):
                # Ensure the mean value matches the column type to avoid dtype incompatibility
                X.loc[X[col] == 0, col] = X[col].dtype.type(self.means[col])
            else:
                # For CuPy arrays, use the dtype of the column (assumes the dtype is uniform for each column)
                X[X[:, col] == 0, col] = self.means[col]

        return X
||||||| c1bda8b
=======
import cudf
import cupy as cp
from sklearn.base import BaseEstimator, TransformerMixin


class ReplaceZeroWithMean(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns  # List of columns to transform

    def fit(self, X, y=None):
        # Compute the mean of non-zero values for each specified column
        self.means = {
            col: (
                X.loc[X[col] != 0, col].mean()  # For cuDF DataFrame
                if isinstance(X, cudf.DataFrame)
                else cp.nanmean(cp.where(X[:, col] != 0, X[:, col], cp.nan))  # For CuPy array
            )
            for col in self.columns
        }
        return self

    def transform(self, X, y=None):
        X = X.copy()  # Make a copy to avoid modifying the original data

        for col in self.columns:
            if isinstance(X, cudf.DataFrame):
                # Ensure the mean value matches the column type to avoid dtype incompatibility
                X.loc[X[col] == 0, col] = X[col].dtype.type(self.means[col])
            else:
                # For CuPy arrays, use the dtype of the column (assumes the dtype is uniform for each column)
                X[X[:, col] == 0, col] = self.means[col]

        return X
>>>>>>> d79c55d92c4bf99c2c8048bbcc5cb2918077f3a5
