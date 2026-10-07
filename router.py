from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class RouteDecision(BaseModel):
    intent: Literal[
        "rag",
        "read_file",
        "validate_file",
        "create_file",
        "edit_file",
        "delete_file",
        "create_folder",
        "delete_folder",
        "out_of_scope",
    ]
    path: str | None = None
    content: str | None = None


class Router:
    """Classify user requests and extract tool arguments."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(RouteDecision)

    async def route(self, text: str) -> RouteDecision:
        prompt = f"""
You are the routing component of Hyperion, an assistant for the HYPER-AI IDE.

Classify the user's request into exactly one intent.

Intents:

- rag:
  Questions about HYPER-AI concepts, documentation,
  architecture, or project knowledge.

- read_file:
  Requests to read, inspect, show, open, view, or understand
  the contents of a workspace file.

- validate_file:
  Requests to validate or check a workspace file.

- create_file:
  Requests to create a new workspace file.

- edit_file:
  Requests to modify, update, or rewrite an existing workspace file.

- delete_file:
  Requests to delete a workspace file.

- create_folder:
  Requests to create a new workspace folder.

- delete_folder:
  Requests to delete an existing workspace folder.

- out_of_scope:
  Requests unrelated to the HYPER-AI project or IDE.


PATH:

For file operations, extract the filename or path mentioned by
the user and preserve it exactly.

A candidate path may be a filename without an extension.

Do not check whether the path exists.
Workspace path resolution is handled separately by the application.

If the user refers only to a generic file description and does not
provide a candidate filename or path, set path to null.

For folder operations, apply the same rule to the folder path.

For rag and out_of_scope, path must be null.


CONTENT:

For create_file and edit_file, extract content explicitly provided
by the user.

Do not invent or modify content.

If the user describes content that should be generated or written,
preserve that request as content.

If no content is provided, set content to null.

For all other intents, content must be null.


IMPORTANT:

Requests that clearly describe an IDE file or folder operation
must be classified according to that operation. Do not classify
a file operation as out_of_scope because the target may not exist.

Do not use workspace existence as a routing criterion.

Return exactly one intent.
Do not invent paths.
Do not invent content.
Preserve user-provided paths and content accurately.

User request:
{text}
""".strip()

        return await self.llm.ainvoke(prompt)

