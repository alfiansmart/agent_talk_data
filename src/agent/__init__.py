"""High level API for the agent package."""

from .datasource import DataSource, join_dataframes
from .analysis import describe, correlation
from .ml import regression, classification, clustering
from .visualization import scatter, line, bar
from .reasoning import (
    ReasoningAgent,
    Tool,
    ToolReasoningAgent,
    openai_llm,
    azure_openai_llm,
)

__all__ = [
    "DataSource",
    "join_dataframes",
    "describe",
    "correlation",
    "regression",
    "classification",
    "clustering",
    "scatter",
    "line",
    "bar",
    "ReasoningAgent",
    "Tool",
    "ToolReasoningAgent",
    "openai_llm",
    "azure_openai_llm",
]
