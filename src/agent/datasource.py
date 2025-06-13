import pandas as pd
from sqlalchemy import create_engine


class DataSource:
    """Handles connections to PostgreSQL, MySQL and reading parquet files."""

    def __init__(self, connection_string: str | None = None, type_: str | None = None):
        """Initialize the data source.

        Parameters
        ----------
        connection_string : str, optional
            SQLAlchemy connection string for the database.
        type_ : str, optional
            Type of the database. Supported values: "postgresql", "mysql".
        """
        self.type = type_
        self.connection_string = connection_string
        self._engine = None
        if connection_string and type_:
            self._engine = create_engine(connection_string)

    def query(self, sql: str) -> pd.DataFrame:
        """Run a SQL query and return a dataframe."""
        if not self._engine:
            raise ValueError("No database connection configured")
        return pd.read_sql(sql, self._engine)

    @staticmethod
    def read_parquet(path: str) -> pd.DataFrame:
        """Read a Parquet file to a dataframe."""
        return pd.read_parquet(path)


def join_dataframes(*dfs: pd.DataFrame, on: str, how: str = "inner") -> pd.DataFrame:
    """Join multiple dataframes on a column."""
    if not dfs:
        raise ValueError("No dataframes provided")
    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on=on, how=how)
    return result
