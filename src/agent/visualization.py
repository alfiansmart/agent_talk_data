import plotly.express as px
import pandas as pd


def scatter(df: pd.DataFrame, x: str, y: str, color: str | None = None):
    fig = px.scatter(df, x=x, y=y, color=color)
    fig.show()
    return fig


def line(df: pd.DataFrame, x: str, y: str, color: str | None = None):
    fig = px.line(df, x=x, y=y, color=color)
    fig.show()
    return fig


def bar(df: pd.DataFrame, x: str, y: str, color: str | None = None):
    fig = px.bar(df, x=x, y=y, color=color)
    fig.show()
    return fig
