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

  Extract the file name or file path exactly as provided by the user.

  If no specific filename or path is provided, path must be null.

- create_file:
  The user wants to create a new file in the IDE workspace.

  Extract the file path only if the user explicitly provides it.

  Extract the requested file content if the user provides or describes
  content that should be written into the new file.

  If the user asks to create a file without specifying content,
  set content to an empty string.

- edit_file:
  The user wants to modify, change, update, or rewrite an existing
  file in the IDE workspace.

  Extract the file path only if the user explicitly provides it.

  Extract the complete new file content if the user provides or
  describes content that should replace the existing file content.

  Do not invent or infer new file content.

  If the user asks to edit a file but does not provide the new content,
  set content to an empty string.

- delete_file:
  The user wants to delete an existing file from the IDE workspace.

  Extract the file path only if the user explicitly provides it.

  Do not infer, guess, autocomplete, or invent the file path.

  If the user asks to delete a file without specifying
  which file, set path to null.

- out_of_scope:
  Requests unrelated to the HYPER-AI project or IDE capabilities.


PATH EXTRACTION RULES:

1. Only set `path` when the user explicitly provides a filename
   or file path in their request.

2. Copy the filename or path exactly as written by the user.

3. Never infer, guess, autocomplete, or invent a filename.

4. Words or phrases such as "my config file", "the deployment file",
   "the YAML file", "the project file", "my Python file",
   "the Python file", or "the file" are NOT filenames.
   In these cases, path must be null.

5. Examples:
   "txt file"
   "the text file"
   "my Python file"
   "my YAML file"
   "my config file"
   "the Python file"
   "the file"
   "my project file"

   These descriptions must always result in path = null.

6. Examples:

   User: "Show me app.yaml"
   -> intent = read_file
   -> path = "app.yaml"

   User: "Open demo/deployment.yaml"
   -> intent = read_file
   -> path = "demo/deployment.yaml"

   User: "What is inside my config file?"
   -> intent = read_file
   -> path = null

   User: "Validate hello/hello.yaml"
   -> intent = validate_file
   -> path = "hello/hello.yaml"

   User: "Can you validate my YAML file?"
   -> intent = validate_file
   -> path = null

   User: "Create hello.py"
   -> intent = create_file
   -> path = "hello.py"
   -> content = ""

   User: "Create hello.py that prints hello world"
   -> intent = create_file
   -> path = "hello.py"
   -> content = the requested code

   User: "Edit hello.py to print hello world"
   -> intent = edit_file
   -> path = "hello.py"
   -> content = "print('hello world')"

7. For rag requests, path must be null.

8. For out_of_scope requests, path must be null.

9. Never invent file content.

10. If no content is specified for create_file or edit_file,
   set content to an empty string.

11. The examples above are instructions for classification only.
    Never copy explanatory text such as "the requested code",
    "the requested content", or "the file content" into the
    `content` field.

12. For delete_file, never invent or infer a path.
    If no specific filename or path is provided, set path to null.

IMPORTANT:
- Return exactly one intent.
- Do not invent paths.
- Do not invent file content.
- Preserve explicitly provided paths exactly.
- Preserve requested code/content accurately.

User request:
{text}
""".strip()

        return await self.llm.ainvoke(prompt)

