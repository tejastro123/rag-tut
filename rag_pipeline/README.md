# Local RAG Pipeline with Ollama + Chroma

## Features

- Local embeddings using Sentence Transformers
- Local LLM using Ollama
- Chroma vector database
- Persistent vector storage
- Clean production-ready structure

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Ollama

Download:
https://ollama.com

### 3. Pull model

```bash
ollama pull llama3
```

### 4. Start Ollama

```bash
ollama serve
```

### 5. Run application

```bash
python app.py
```
