from rag_pipeline import RAGPipeline

KNOWLEDGE_BASE = """
LangChain is a framework for building LLM applications.

LangGraph is used for stateful AI workflows.

Chroma is a vector database.

Ollama allows running local LLMs.
"""

rag = RAGPipeline()

rag.ingest_documents([KNOWLEDGE_BASE])

questions = [
    "What is LangChain?",
    "What is LangGraph?",
    "What is Ollama?",
    "Who created Python?",
]

for q in questions:
    print("=" * 50)
    print(f"Q: {q}")

    answer = rag.ask(q)

    print(f"A: {answer}")
