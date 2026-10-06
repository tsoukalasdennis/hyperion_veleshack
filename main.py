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
from helpers import (
    ReadFileError,
    ValidateFileError,
    read_file,
    validate_file,
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
    decision = await router.route(request.text)

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
                yield f"data: {json.dumps({'response': chunk.text})}\n\n"

    elif decision.intent == "read_file":
        if decision.path is None:
            response = "Which file would you like me to read?"
            yield f"data: {json.dumps({'response': response})}\n\n"

        else:
            try:
                content = await read_file(decision.path)

                prompt = f"""
You are Hyperion, an assistant for the HYPER-AI IDE.

The user asked you to inspect a file from the IDE workspace.

File path:
{decision.path}

File content:
{content}

Answer the user's original request using the file content.

Be concise and clear.
Do not invent information that is not present in the file.

User request:
{request.text}
""".strip()

                async for chunk in llm.astream(prompt):
                    if chunk.text:
                        yield f"data: {json.dumps({'response': chunk.text})}\n\n"

            except ReadFileError as exc:
                response = str(exc)

                yield f"data: {json.dumps({'response': response})}\n\n"

    elif decision.intent == "validate_file":
        if decision.path is None:
            response = "Which file would you like me to validate?"
            yield f"data: {json.dumps({'response': response})}\n\n"

        else:
            try:
                report = await validate_file(decision.path)

                prompt = f"""
You are Hyperion, an assistant for the HYPER-AI IDE.

The user asked you to validate a file from the IDE workspace.

File path:
{decision.path}

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
                        yield f"data: {json.dumps({'response': chunk.text})}\n\n"

            except ValidateFileError as exc:
                response = str(exc)

                yield f"data: {json.dumps({'response': response})}\n\n"

    elif decision.intent == "create_file":
        if decision.path is None:
            response = "Which file would you like me to create?"
            yield f"data: {json.dumps({'response': response})}\n\n"

        else:
            action = {
                "action": "create_file",
                "path": decision.path,
                "content": decision.content or "",
            }

            yield f"data: {json.dumps(action)}\n\n"

            response = f"Created {decision.path}."
            yield f"data: {json.dumps({'response': response})}\n\n"

    else:
        response = (
            "I can help with HYPER-AI project documentation "
            "and IDE tasks."
        )

        yield f"data: {json.dumps({'response': response})}\n\n"

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
