import sys
from pathlib import Path

# Ensure src is on path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src/agent'))

from datasource import DataSource  # noqa: E402


def test_query_with_connection_string_only():
    ds = DataSource('sqlite://')
    df = ds.query('SELECT 1 AS a')
    assert df.iloc[0]['a'] == 1
