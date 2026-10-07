import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from rag_langchain import RAG
from router import Router
from memory import ConversationMemory
from helpers import (
    ReadFileError,
    ValidateFileError,
    read_file,
    validate_file,
    resolve_existing_path,
    ResolvePathError,
    find_matching_paths,
)


load_dotenv()


API_KEY = os.environ.get("API_KEY", "")
BASE_URL = "https://legion1.di.uoa.gr/v1"
MODEL = "llama3.1"


llm = ChatOpenAI(
    model=MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    max_completion_tokens=2048,
)


router = Router(llm)
memory = ConversationMemory(max_turns=10)


async def generate_file_content(instruction: str) -> str:
    prompt = f"""
You are Hyperion, an AI coding assistant.

Generate the actual content that should be written into a workspace file.

Follow the user's instruction exactly.

Do not explain what you are doing.
Do not wrap the result in Markdown fences unless the user explicitly
asks for them.

User instruction:
{instruction}
""".strip()

    response = await llm.ainvoke(prompt)

    return response.text.strip()


app = FastAPI(title="Hyperion Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_id: str
    text: str


print("Building RAG index...")
RAG_INDEX = RAG()
print("RAG index ready.")


def build_prompt(question: str, results) -> str:
    context_parts = []

    for result in results:
        context_parts.append(
            f"Source: {result.metadata['source']}\n"
            f"{result.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    return f"""
You are Hyperion, an assistant for the HYPER-AI project.

Answer the user's question using the provided HYPER-AI documentation.

Rules:
- Prefer information from the provided documentation.
- Do not invent facts that are not supported by the documentation.
- If the documentation does not contain enough information to answer the question,
  say that the available documentation does not provide enough information.
- Answer clearly and concisely.

HYPER-AI DOCUMENTATION:

{context}

USER QUESTION:

{question}
""".strip()


async def generate_reply(request: ChatRequest):
    conversation_history = memory.format_history(
        request.user_id
    )

    decision = await router.route(
        request.text,
        conversation_history=conversation_history,
    )

    assistant_response = ""

    if decision.intent == "rag":
        results = RAG_INDEX.search(
            request.text,
            top_k=3,
        )

        prompt = build_prompt(
            request.text,
            results,
        )

        async for chunk in llm.astream(prompt):
            if chunk.text:
                assistant_response += chunk.text

                yield f"data: {json.dumps({'response': chunk.text})}\n\n"

    elif decision.intent == "read_file":
        if decision.path is None:
            response = "Which file would you like me to read?"
            assistant_response = response

            yield f"data: {json.dumps({'response': response})}\n\n"

        else:
            try:
                resolved_path = await resolve_existing_path(
                    decision.path
                )

                content = await read_file(
                    resolved_path
                )

                if not content.strip():
                    response = f"The file `{resolved_path}` is empty."
                else:
                    response = (
                        f"The file `{resolved_path}` contains:\n\n"
                        f"```text\n{content}\n```"
                    )

                assistant_response = response

                yield f"data: {json.dumps({'response': response})}\n\n"

            except (ResolvePathError, ReadFileError) as exc:
                response = str(exc)
                assistant_response = response

                yield f"data: {json.dumps({'response': response})}\n\n"

    elif decision.intent == "validate_file":
        if decision.path is None:
            response = "Which file would you like me to validate?"
            assistant_response = response

            yield f"data: {json.dumps({'response': response})}\n\n"

        else:
            try:
                resolved_path = await resolve_existing_path(
                    decision.path
                )

                report = await validate_file(
                    resolved_path
                )

                prompt = f"""
You are Hyperion, an assistant for the HYPER-AI IDE.

The user asked you to validate a file from the IDE workspace.

File path:
{resolved_path}

Validation report:
{report}

Explain the validation result to the user clearly.

Be concise and clear.
Do not invent validation results that are not present in the report.

User request:
{request.text}
""".strip()

                async for chunk in llm.astream(prompt):
                    if chunk.text:
                        assistant_response += chunk.text

                        yield f"data: {json.dumps({'response': chunk.text})}\n\n"

            except (ResolvePathError, ValidateFileError) as exc:
                response = str(exc)
                assistant_response = response

                yield f"data: {json.dumps({'response': response})}\n\n"

    elif decision.intent == "create_file":
        if decision.path is None:
            response = "Which file would you like me to create?"
            assistant_response = response

            yield f"data: {json.dumps({'response': response})}\n\n"

        else:
            matches = await find_matching_paths(
                decision.path
            )

            if len(matches) == 1:
                response = f"The file `{matches[0]}` already exists."
                assistant_response = response

                yield f"data: {json.dumps({'response': response})}\n\n"

            elif len(matches) > 1:
                response = (
                    f"Several files named `{decision.path}` already exist: "
                    f"{', '.join(matches)}. "
                    "Please specify the full path."
                )
                assistant_response = response

                yield f"data: {json.dumps({'response': response})}\n\n"

            else:
                content_decision = await router.interpret_content(
                    user_text=request.text,
                    intent=decision.intent,
                    path=decision.path,
                )

                if content_decision.mode == "clarify":
                    response = (
                        content_decision.clarification
                        or "Could you clarify what content the file should contain?"
                    )
                    assistant_response = response

                    yield f"data: {json.dumps({'response': response})}\n\n"

                elif content_decision.mode == "empty":
                    action = {
                        "action": "create_file",
                        "path": decision.path,
                        "content": "",
                    }

                    yield f"data: {json.dumps(action)}\n\n"

                    response = f"Created {decision.path}."
                    assistant_response = response

                    yield f"data: {json.dumps({'response': response})}\n\n"

                elif content_decision.mode == "literal":
                    content = content_decision.literal_content or ""

                    action = {
                        "action": "create_file",
                        "path": decision.path,
                        "content": content,
                    }

                    yield f"data: {json.dumps(action)}\n\n"

                    response = f"Created {decision.path}."
                    assistant_response = response

                    yield f"data: {json.dumps({'response': response})}\n\n"

                elif content_decision.mode == "generate":
                    content = await generate_file_content(
                        content_decision.generation_instruction or ""
                    )

                    action = {
                        "action": "create_file",
                        "path": decision.path,
                        "content": content,
                    }

                    yield f"data: {json.dumps(action)}\n\n"

                    response = f"Created {decision.path}."
                    assistant_response = response

                    yield f"data: {json.dumps({'response': response})}\n\n"

    elif decision.intent == "edit_file":
        if decision.path is None:
            response = "Which file would you like me to edit?"
            assistant_response = response

            yield f"data: {json.dumps({'response': response})}\n\n"

        else:
            try:
                resolved_path = await resolve_existing_path(
                    decision.path
                )

                action = {
                    "action": "edit_file",
                    "path": resolved_path,
                    "content": decision.content or "",
                }

                yield f"data: {json.dumps(action)}\n\n"

                response = f"Updated {resolved_path}."
                assistant_response = response

                yield f"data: {json.dumps({'response': response})}\n\n"

            except ResolvePathError as exc:
                response = str(exc)
                assistant_response = response

                yield f"data: {json.dumps({'response': response})}\n\n"

    elif decision.intent == "delete_file":
        if decision.path is None:
            response = "Which file would you like me to delete?"
            assistant_response = response

            yield f"data: {json.dumps({'response': response})}\n\n"

        else:
            try:
                resolved_path = await resolve_existing_path(
                    decision.path
                )

                action = {
                    "action": "delete_file",
                    "path": resolved_path,
                }

                yield f"data: {json.dumps(action)}\n\n"

                response = f"Deleted {resolved_path}."
                assistant_response = response

                yield f"data: {json.dumps({'response': response})}\n\n"

            except ResolvePathError as exc:
                response = str(exc)
                assistant_response = response

                yield f"data: {json.dumps({'response': response})}\n\n"

    elif decision.intent == "create_folder":
        if decision.path is None:
            response = "Which folder would you like me to create?"
            assistant_response = response

            yield f"data: {json.dumps({'response': response})}\n\n"

        else:
            action = {
                "action": "create_folder",
                "path": decision.path,
            }

            yield f"data: {json.dumps(action)}\n\n"

            response = f"Created folder {decision.path}."
            assistant_response = response

            yield f"data: {json.dumps({'response': response})}\n\n"

    elif decision.intent == "delete_folder":
        if decision.path is None:
            response = "Which folder would you like me to delete?"
            assistant_response = response

            yield f"data: {json.dumps({'response': response})}\n\n"

        else:
            action = {
                "action": "delete_folder",
                "path": decision.path,
            }

            yield f"data: {json.dumps(action)}\n\n"

            response = f"Deleted folder {decision.path}."
            assistant_response = response

            yield f"data: {json.dumps({'response': response})}\n\n"

    else:
        response = (
            "I can help with HYPER-AI project documentation "
            "and IDE tasks."
        )

        assistant_response = response

        yield f"data: {json.dumps({'response': response})}\n\n"

    memory.add_user_message(
        request.user_id,
        request.text,
    )

    memory.add_assistant_message(
        request.user_id,
        assistant_response,
    )

    yield "data: [DONE]\n\n"


@app.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(
        generate_reply(request),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )