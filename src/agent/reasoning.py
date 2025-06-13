from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any, Dict


@dataclass
class ReasoningAgent:
    """Very simple reasoning agent that can use an LLM backend."""

    llm_func: Callable[[str], str]

    def ask(self, question: str) -> str:
        """Generate an answer using the provided llm function."""
        return self.llm_func(question)


@dataclass
class Tool:
    """Represents a callable tool for the agent."""

    name: str
    func: Callable[..., Any]
    description: str = ""


@dataclass
class ToolReasoningAgent:
    """Reasoning agent that can call tools suggested by the LLM."""

    llm_func: Callable[[str], str]
    tools: Dict[str, Tool]

    def ask(self, question: str) -> Any:
        """Ask a question and execute the tool chosen by the LLM."""
        tool_desc = "\n".join(
            f"{name}: {tool.description}" for name, tool in self.tools.items()
        )
        prompt = (
            "You are a data assistant. Available tools:\n" + tool_desc + "\n" +
            f"Question: {question}\n" +
            "Respond with JSON {\"tool\": \"name\", \"args\": {...}}"
        )
        response = self.llm_func(prompt)
        try:
            import json

            data = json.loads(response)
            tool_name = data.get("tool")
            args = data.get("args", {})
            if tool_name not in self.tools:
                raise ValueError(f"Unknown tool {tool_name}")
            return self.tools[tool_name].func(**args)
        except Exception as exc:  # pragma: no cover - response parsing
            raise RuntimeError(
                f"Could not use tool from response: {response}"
            ) from exc


# Example usage with OpenAI (requires openai package and API key)
try:
    import openai

    def openai_llm(prompt: str) -> str:
        completion = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=200)
        return completion.choices[0].text.strip()
except Exception:  # pragma: no cover - openai not installed during tests
    def openai_llm(prompt: str) -> str:
        raise RuntimeError("OpenAI is not available")


# Azure OpenAI helper
try:  # OpenAI >= 1.0 style client
    from openai import AzureOpenAI  # type: ignore
    import os

    _azure_client = AzureOpenAI(
        api_version="2023-07-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_KEY", ""),
    )

    def azure_openai_llm(prompt: str) -> str:
        """Call an Azure OpenAI deployment (defaults to o3-mini)."""
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "o3-mini")
        completion = _azure_client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return completion.choices[0].message.content.strip()
except Exception:
    try:  # fallback to legacy client
        import openai  # type: ignore
        import os

        def azure_openai_llm(prompt: str) -> str:
            openai.api_type = "azure"
            openai.api_version = "2023-07-01-preview"
            openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT", "")
            openai.api_key = os.getenv("AZURE_OPENAI_KEY", "")
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "o3-mini")
            completion = openai.ChatCompletion.create(
                deployment_id=deployment,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            return completion.choices[0].message["content"].strip()
    except Exception:  # pragma: no cover - openai not installed
        def azure_openai_llm(prompt: str) -> str:
            raise RuntimeError("Azure OpenAI is not available")
