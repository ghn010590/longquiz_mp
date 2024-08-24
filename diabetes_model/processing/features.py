import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin



class Mapper(BaseEstimator, TransformerMixin):
    def __init__(self, variable, mappings):
        self.variable = variable
        self.mappings = mappings

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure X is a DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.copy()
        elif isinstance(X, np.ndarray):
            # Assuming X was converted to ndarray by previous transformers, convert it back to DataFrame
            # This assumes that you know the column names; if not, additional handling is needed
            X = pd.DataFrame(X, columns=[self.variable] + ['other_columns_if_known'])

        # Map the categorical variable to new codes
        X[self.variable] = X[self.variable].map(self.mappings)

        # Fill NA values with -1 and convert to integer
        X[self.variable] = X[self.variable].fillna(-1).astype(int)

        # Print the first few rows to debug
        print(X.head())

        return X

