import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from docx import Document


load_dotenv()

DOCS_DIR = Path("docs")

EMBEDDING_URL = "https://legion1.di.uoa.gr/v1/embeddings"
EMBEDDING_MODEL = "nomic-embed-text"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_documents() -> list[dict]:
    documents = []

    for path in DOCS_DIR.glob("*.docx"):
        doc = Document(path)

        text = "\n".join(
            paragraph.text.strip()
            for paragraph in doc.paragraphs
            if paragraph.text.strip()
        )

        documents.append(
            {
                "source": path.name,
                "text": text,
            }
        )

    return documents


def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def build_chunks() -> list[dict]:
    chunks = []

    for document in load_documents():
        for index, chunk in enumerate(chunk_text(document["text"])):
            chunks.append(
                {
                    "source": document["source"],
                    "chunk_id": index,
                    "text": chunk,
                }
            )

    return chunks


def embed(text: str) -> list[float]:
    response = httpx.post(
        EMBEDDING_URL,
        headers={
            "Authorization": f"Bearer {os.environ['API_KEY']}",
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
    dot_product = sum(x * y for x, y in zip(a, b))

    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(y * y for y in b) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def build_index(chunks: list[dict]) -> list[dict]:
    index = []

    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i + 1}/{len(chunks)}...")

        index.append(
            {
                **chunk,
                "embedding": embed(chunk["text"]),
            }
        )

    return index


def search_docs(
    query: str,
    index: list[dict],
    top_k: int = 3,
) -> list[dict]:

    query_embedding = embed(query)

    results = []

    for item in index:
        score = cosine_similarity(
            query_embedding,
            item["embedding"],
        )

        results.append(
            {
                "source": item["source"],
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "score": score,
            }
        )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:top_k]


if __name__ == "__main__":
    chunks = build_chunks()

    print(f"Loaded {len(chunks)} chunks")

    index = build_index(chunks)

    query = "What are HYPER-AI Open Connectors?"

    print(f"\nSearching for: {query}\n")

    results = search_docs(query, index)

    for result in results:
        print("=" * 80)
        print(f"Score: {result['score']:.4f}")
        print(f"Source: {result['source']}")
        print(f"Chunk: {result['chunk_id']}")
        print(result["text"][:500])
