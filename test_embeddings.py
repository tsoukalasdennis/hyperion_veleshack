import json
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["API_KEY"]
BASE_URL = "https://legion1.di.uoa.gr/v1"

response = httpx.post(
    f"{BASE_URL}/embeddings",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": "nomic-embed-text",
        "input": "What are HYPER-AI Open Connectors?",
    },
    timeout=30,
)

print("Status:", response.status_code)

if response.status_code != 200:
    print(response.text)
    raise SystemExit(1)

data = response.json()

embedding = data["data"][0]["embedding"]

print("Embedding dimensions:", len(embedding))
print("First 10 values:", embedding[:10])
