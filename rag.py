import os
from pathlib import Path

import httpx
from docx import Document
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["API_KEY"]
BASE_URL = "https://legion1.di.uoa.gr/v1"
EMBEDDING_MODEL = "nomic-embed-text"

DOCS_DIR = Path("docs")
CHUNK_SIZE = 1000


def load_documents() -> list[dict]:
    """Load all DOCX documents from the docs directory."""

    documents = []

    for path in sorted(DOCS_DIR.glob("*.docx")):
        document = Document(path)

        paragraphs = [
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        text = "\n".join(paragraphs)

        documents.append(
            {
                "source": path.name,
                "text": text,
            }
        )

    return documents


def chunk_text(text: str) -> list[str]:
    """Split text into paragraph-aware chunks."""

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n")
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        candidate = (
            paragraph
            if not current_chunk
            else f"{current_chunk}\n{paragraph}"
        )

        if len(candidate) <= CHUNK_SIZE:
            current_chunk = candidate
            continue

        if current_chunk:
            chunks.append(current_chunk)

        if len(paragraph) > CHUNK_SIZE:
            start = 0

            while start < len(paragraph):
                end = start + CHUNK_SIZE
                chunks.append(paragraph[start:end].strip())
                start = end

            current_chunk = ""
        else:
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def build_chunks(documents: list[dict]) -> list[dict]:
    """Turn documents into searchable chunks with metadata."""

    chunks = []

    for document in documents:
        text_chunks = chunk_text(document["text"])

        for chunk_id, text in enumerate(text_chunks):
            chunks.append(
                {
                    "source": document["source"],
                    "chunk_id": chunk_id,
                    "text": text,
                }
            )

    return chunks


def embed(text: str) -> list[float]:
    """Create an embedding using the HYPER-AI embedding endpoint."""

    response = httpx.post(
        f"{BASE_URL}/embeddings",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": EMBEDDING_MODEL,
            "input": text,
        },
        timeout=30,
    )

    response.raise_for_status()

    return response.json()["data"][0]["embedding"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""

    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def build_index(chunks: list[dict]) -> list[dict]:
    """Embed every chunk and build the in-memory search index."""

    index = []

    for number, chunk in enumerate(chunks, start=1):
        print(f"Embedding chunk {number}/{len(chunks)}...")

        vector = embed(chunk["text"])

        index.append(
            {
                **chunk,
                "embedding": vector,
            }
        )

    return index


def search_docs(
    query: str,
    index: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Find the most relevant document chunks for a query."""

    query_embedding = embed(query)

    scored_results = []

    for item in index:
        score = cosine_similarity(
            query_embedding,
            item["embedding"],
        )

        scored_results.append(
            {
                "source": item["source"],
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "score": score,
            }
        )

    scored_results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return scored_results[:top_k]


def build_rag_index() -> list[dict]:
    """Load documents, chunk them, and build the searchable index."""

    documents = load_documents()
    chunks = build_chunks(documents)

    print(f"Loaded {len(documents)} documents")
    print(f"Created {len(chunks)} chunks")

    return build_index(chunks)


if __name__ == "__main__":
    index = build_rag_index()

    query = "What are HYPER-AI Open Connectors?"

    print(f"\nSearching for: {query}\n")

    results = search_docs(query, index)

    for result in results:
        print(
            f"Score {result['score']:.4f} "
            f"{result['source']} "
            f"chunk {result['chunk_id']}"
        )

        print(result["text"][:500])
        print()
