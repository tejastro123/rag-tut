# Local Production AI System

## Install Ollama

Download:
<https://ollama.com/download>

Pull model:

```bash
ollama pull llama3
```

Start Ollama:

```bash
ollama serve
```

---

## Install Dependencies

```bash
uv venv
uv pip install -r requirements.txt
```

---

## Run Application

```bash
uv run app/main.py
```

---

## Features

* Local LLMs with Ollama
* Chroma vector database
* HuggingFace embeddings
* Persistent caching
* Token budgeting
* Cost optimization
* Production-ready structure
* RAG pipeline

---

## Future Improvements

* Redis cache
* Async support
* FastAPI server
* LangSmith tracing
* Hybrid search
* Reranking
* Multi-agent workflows
