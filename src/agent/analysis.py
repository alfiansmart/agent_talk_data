import pandas as pd


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Return basic statistics for the dataframe."""
    return df.describe(include='all')


def correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix."""
    return df.corr(numeric_only=True)
