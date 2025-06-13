"""Example usage of the agent package."""
from agent import (
    DataSource,
    join_dataframes,
    describe,
    regression,
    scatter,
    Tool,
    ToolReasoningAgent,
    azure_openai_llm,
)


def main():
    # Example using parquet file
    sales = DataSource.read_parquet("data/sales.parquet")

    print(describe(sales))

    model, mse = regression(sales, target="revenue")
    print("Regression MSE:", mse)

    scatter(sales, x="date", y="revenue")

    # Tool reasoning agent using Azure OpenAI
    tools = {
        "describe": Tool(
            name="describe",
            func=lambda: describe(sales),
            description="Show dataframe statistics",
        ),
        "plot": Tool(
            name="plot",
            func=lambda: scatter(sales, x="date", y="revenue"),
            description="Plot revenue over time",
        ),
    }
    agent = ToolReasoningAgent(azure_openai_llm, tools)
    try:
        answer = agent.ask("Show me the stats")
        print(answer)
    except RuntimeError as err:
        print(err)


if __name__ == "__main__":
    main()
