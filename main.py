import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from rag import build_rag_index, search_docs

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
RAG_INDEX = build_rag_index()
print("RAG index ready.")


def build_prompt(question: str, results: list[dict]) -> str:
    context_parts = []

    for result in results:
        context_parts.append(
            f"Source: {result['source']}\n"
            f"{result['text']}"
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
    results = search_docs(
        request.text,
        RAG_INDEX,
        top_k=3,
    )

    prompt = build_prompt(
        request.text,
        results,
    )

    async for chunk in llm.astream(prompt):
        if chunk.text:
            yield f"data: {json.dumps({'response': chunk.text})}\n\n"

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
