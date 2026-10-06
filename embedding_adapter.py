import os

import httpx
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv()

API_KEY = os.environ["API_KEY"]
BASE_URL = "https://legion1.di.uoa.gr/v1"
EMBEDDING_MODEL = "nomic-embed-text"


class NomicEmbeddings(Embeddings):
    """LangChain adapter for the HYPER-AI embedding endpoint."""

    def _embed(self, text: str) -> list[float]:
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

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
