import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns  
        self.encoders = {}  

    def fit(self, x: pd.DataFrame, y: pd.Series = None) -> 'MultiColumnLabelEncoder':
        self.columns = x.columns if self.columns is None else self.columns
        for col in self.columns:
            if col in x:
                le = LabelEncoder()
                le.fit(x[col].astype(str).fillna(''))
                self.encoders[col] = le
            else:
                raise ValueError(f"Column '{col}' not found in the DataFrame")
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform the DataFrame using the fitted encoders."""
        df = x.copy()  # Make a copy of the DataFrame to avoid modifying the original data
        for col in self.columns:
            df[col] = self.encoders[col].transform(df[col].astype(str).fillna(''))  # Apply encoding
        return df

    def fit_transform(self, x: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        self.fit(x, y)
        return self.transform(x)

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform the encoded DataFrame back to the original values."""
        df = x.copy()  # Make a copy to avoid modifying the original DataFrame
        for col in self.columns:
            df[col] = self.encoders[col].inverse_transform(df[col])  # Use inverse transform on each column
        return df
