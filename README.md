# agent_talk_data

`agent_talk_data` provides building blocks for an AI agent that can query
multiple data sources, perform analysis, run simple machine learning models and
visualise results. It also includes reasoning helpers that can be wired to a
large language model service such as OpenAI or Azure OpenAI.

## Features

- Query PostgreSQL or MySQL via SQLAlchemy
- Read `.parquet` files with pandas
- Join data from different sources
- Compute statistics and correlations
- Run linear/logistic regression and k-means clustering
- Plot using Plotly
- Simple reasoning layer with pluggable LLM backend and basic tool calling

## Example

```bash
pip install pandas sqlalchemy scikit-learn plotly openai
python examples/demo.py
```

See `examples/demo.py` for a short demonstration. The reasoning example works
with Azure OpenAI by setting `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY` and
`AZURE_OPENAI_DEPLOYMENT` environment variables. The helper uses `from openai
import AzureOpenAI` when available and falls back to the legacy client
configuration.

## Development

A list of open tasks can be found in `TODO.md`.
