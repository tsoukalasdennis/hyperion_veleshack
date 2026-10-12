# Hyperion

**Hyperion** is an LLM-powered agentic assistant for the **HYPER-AI IDE**, developed for **VelesHack 2026**.

It allows users to interact with the IDE through natural language, combining **RAG**, **session memory**, and **IDE file operations**.

## Features

*  **RAG** — answers HYPER-AI questions using project documentation.
*  **Session Memory** — maintains context across conversations.
*  **IDE Actions** — create, read, edit and delete files and folders.
*  **Semantic Editing** — generate file content from natural-language instructions while preserving explicitly provided content.
*  **SSE Streaming** — streams responses and IDE actions to the frontend.
*  **Guardrails** — rejects unrelated requests.
*  **Dockerized** — ready for deployment through the challenge's Docker interface.

## Architecture

<img width="349" height="666" alt="Στιγμιότυπο οθόνης 2026-10-07 170432" src="https://github.com/user-attachments/assets/0e0d23b7-a735-4992-90b2-58c675b2b41d" />


## Tech Stack

* Python
* FastAPI
* LangChain
* FAISS
* Pydantic
* Docker
* `llama3.1` LLM
* `nomic-embed-text` embeddings

## Running locally

Start the agent with:

```bash
uv run main.py
```

Or with Docker:

```bash
docker compose up --build
```

The service listens on:

```text
http://localhost:8000
```

### Docker Hub

```text
tsoukalasdennis/hyperion:latest
```

Run with:

```bash
docker run --env-file .env -p 8000:8000 \
  tsoukalasdennis/hyperion:latest
```


## What I learned

VelesHack 2026 was my **first hackathon**. Building Hyperion was a great learning
experience, and seeing other projects gave me a clearer idea of what I would do
differently if I started again.

The main areas I would improve are:

* **RAG and source attribution** — make retrieval more reliable and show the
  documentation sources behind each answer, so users can check where information
  came from.
* **IDE actions** — redesign how actions are represented and executed instead of
  relying on loosely connected instructions. I would use explicit, structured
  actions and validate generated or edited files with the IDE before reporting
  success.
* **Action orchestration** — introduce workers that can run independent tasks in
  parallel and dependent tasks in sequence, with clear results and error handling
  at each step.
* **Safety for destructive actions** — ask the user to confirm before deleting
  files or folders.
* **Less prompt-only behavior** — move important rules and checks into
  deterministic code where possible, keeping prompts focused on the parts that
  genuinely need language-model judgment.

The guardrails and session memory felt like solid foundations, so I would keep
them and focus on testing them alongside the larger changes. These are lessons
and ideas for a future version, not claims that those improvements are already
implemented.


## Project

**Challenge:** VelesHack 2026

**Category:** Challenge 1 (HYPER-AI): Hyperion - An LLM-Powered Agentic Assistant

**Project name:** HyperAgents

## License

[Apache 2.0](LICENCE)
