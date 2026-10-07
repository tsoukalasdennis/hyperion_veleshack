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

    content_request: str | None = None

    needs_clarification: bool = False

    clarification: str | None = None


class ContentDecision(BaseModel):
    mode: Literal[
        "empty",
        "literal",
        "generate",
        "clarify",
    ]

    literal_content: str | None = None
    generation_instruction: str | None = None
    clarification: str | None = None


class Router:
    """Classify user requests and extract tool arguments."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(RouteDecision)
        self.content_llm = llm.with_structured_output(ContentDecision)

    async def route(
        self,
        text: str,
        conversation_history: str = "",
    ) -> RouteDecision:
        if conversation_history:
            history_section = conversation_history
        else:
            history_section = "(No previous conversation.)"

        prompt = f"""
You are the routing component of Hyperion, an assistant for the HYPER-AI IDE.

Classify the user's request into exactly one intent.

You have access to the previous conversation for this session.

Use the previous conversation to resolve references such as:
- "it"
- "that file"
- "that folder"
- "the file we just created"
- "the one from before"
- "open it"
- "edit it"
- "delete it"

When the current request refers to something from the previous conversation,
use the previous conversation to identify the relevant file or folder path.

The current user request is the source of truth for the new action.
Do not invent a path when the previous conversation does not provide enough
information to resolve the reference safely.

If a reference cannot be resolved from the conversation, set path to null
and use needs_clarification when appropriate.

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

If the user refers to a file indirectly, use the previous conversation
to resolve the referenced path.

For example:

Previous conversation:
USER: Create config.py
ASSISTANT: Created config.py.

Current request:
Read it

The correct path is:
config.py

Do not check whether the path exists.
Workspace path resolution is handled separately by the application.

If the user refers only to a generic file description and the path cannot
be resolved from the current request or previous conversation, set path to null.

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

Do not use the presence or absence of file content to decide
whether a request is a file operation.

Use conversation history only to resolve references and maintain context.

Return exactly one intent.
Do not invent paths.
Do not invent content.
Preserve user-provided paths and content accurately.

PREVIOUS CONVERSATION:

{history_section}

CURRENT USER REQUEST:

{text}
""".strip()

        return await self.llm.ainvoke(prompt)

    async def interpret_content(
        self,
        user_text: str,
        intent: str,
        path: str | None,
    ) -> ContentDecision:
        prompt = f"""
You are the content interpretation component of an AI coding assistant.

Determine what the user means about the content of a workspace file.

You must choose exactly one mode:

empty
The user wants the file to contain no content.

Choose empty when:
- the user asks to create a file without specifying any content;
- the user explicitly says the file should be empty;
- the user says the file should contain no content.

Do not choose clarify merely because no content was provided.

literal
The user provided the exact content that should be written to the file.
Return that content exactly in literal_content.
Do not rewrite it, explain it, or generate anything.

generate
The user described what the file should contain and expects the assistant
to generate the content.
Return the user's requested content description in generation_instruction.
Do not generate the final file content.

clarify
You cannot safely determine whether the user supplied exact content
or requested generated content, and the request is genuinely ambiguous.

Important rules:

- Prefer empty when the user requests a file but provides no content.
- Treat explicit requests for an empty file as empty.
- Never guess between literal and generate when the meaning is genuinely unclear.
- Never generate file content yourself.
- Preserve literal content exactly.
- Do not put the same information into multiple fields.
- For empty, all content fields must be null.
- For literal, only literal_content may be populated.
- For generate, only generation_instruction may be populated.
- For clarify, only clarification may be populated.
- The file path is metadata, not file content.
- The original user request is the source of truth.

Intent:
{intent}

Target path:
{path}

Original user request:
{user_text}
""".strip()

        return await self.content_llm.ainvoke(prompt)