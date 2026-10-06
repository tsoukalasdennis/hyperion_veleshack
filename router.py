from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class RouteDecision(BaseModel):
    intent: Literal["rag", "read_file","validate_file", "out_of_scope"]
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
  The user wants to read, inspect, show, open, view, or ask
  what is inside a file from the IDE workspace.

  This includes requests such as:
  "What is inside my config file?"
  "Show me the deployment file"
  "Can you inspect my YAML file?"

  Even when no exact filename is provided, classify the request
  as read_file and set path to null.

- validate_file:
  The user wants to validate, check, verify, or inspect whether
  a file is valid according to the IDE's validation rules.

  For validate_file requests, extract the file name or file path
  exactly as provided by the user.

  If no specific filename or path is provided, path must be null.

- out_of_scope:
  Requests unrelated to the HYPER-AI project or IDE capabilities.

PATH EXTRACTION RULES:

1. Only set `path` when the user explicitly provides a filename
   or file path in their request.

2. Copy the filename or path exactly as written by the user.

3. Never infer, guess, autocomplete, or invent a filename.

4. Words such as "my config file", "the deployment file",
   "the YAML file", or "the project file" are NOT filenames.
   In these cases, path must be null.

5. For example:

   User: "Show me app.yaml"
   -> path = "app.yaml"

   User: "Open demo/deployment.yaml"
   -> path = "demo/deployment.yaml"

   User: "What is inside my config file?"
   -> path = null

   User: "Inspect the deployment file"
   -> path = null

   User: "Validate hello/hello.yaml"
   -> intent = validate_file
   -> path = "hello/hello.yaml"

   User: "Is docker-compose.yml valid?"
   -> intent = validate_file
   -> path = "docker-compose.yml"

   User: "Can you validate my YAML file?"
   -> intent = validate_file
   -> path = null

6. For rag requests, path must be null.

7. For out_of_scope requests, path must be null.

User request:
{text}
""".strip()
        return await self.llm.ainvoke(prompt)
