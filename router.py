from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class RouteDecision(BaseModel):
    intent: Literal["rag", "read_file", "out_of_scope"]
    path: str | None = None


class Router:
    """Classify user requests and extract tool arguments."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(RouteDecision)

    async def route(self, text: str) -> RouteDecision:
        prompt = f"""
You are the routing component of Hyperion, an assistant for the HYPER-AI IDE.

Classify the user's request into exactly one of these intents:

- rag:
  Questions about HYPER-AI concepts, architecture, documentation,
  or project knowledge.

- read_file:
  The user wants to read, inspect, show, open, or view a file
  from the IDE workspace.

- out_of_scope:
  Requests unrelated to the HYPER-AI project or IDE capabilities.

For read_file requests, extract the file name or file path exactly
as provided by the user.

Important:
- Do not invent a file path.
- Do not interpret or replace the filename with something else.
- If the user asks for "app.yaml", the path must be "app.yaml".
- If the user asks for "demo/deployment.yaml", the path must be
  "demo/deployment.yaml".
- If no specific file is provided, path must be null.
- For rag requests, path must be null.
- For out_of_scope requests, path must be null.

User request:
{text}
""".strip()

        return await self.llm.ainvoke(prompt)
