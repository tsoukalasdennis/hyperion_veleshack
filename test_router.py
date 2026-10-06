import asyncio
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from router import Router


load_dotenv()


llm = ChatOpenAI(
    model="llama3.1",
    base_url="https://legion1.di.uoa.gr/v1",
    api_key=os.environ["API_KEY"],
    max_completion_tokens=100,
)


router = Router(llm)


async def main():
    questions = [
        "What is the HYPER-AI Resource Model?",
        "Show me app.yaml",
        "Can you inspect demo/deployment.yaml?",
        "Open docker-compose.yml",
        "What is inside my config file?",
        "What is the weather today?",
    ]

    for question in questions:
        decision = await router.route(question)

        print(f"Question: {question}")
        print(f"Intent:   {decision.intent}")
        print(f"Path:     {decision.path}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
