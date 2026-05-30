# advanced_rag_production.py

"""
Production-Ready Advanced RAG System
====================================

Features:

* Local LLMs with Ollama
* Hybrid Retrieval (BM25 + Vector)
* Multi-Query Retrieval
* Parent Document Retrieval
* Context Compression
* Persistent ChromaDB
* Structured Logging
* Error Handling
* Configurable Settings
* Production Architecture

Requirements:
uv add langchain
uv add langchain-community
uv add langchain-chroma
uv add langchain-ollama
uv add chromadb
uv add rank-bm25
uv add python-dotenv

Pull Ollama Models:
ollama pull llama3
ollama pull nomic-embed-text
"""

from **future** import annotations

import os
import logging
import uuid
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma

from langchain_community.retrievers import BM25Retriever

from langchain.retrievers import (
EnsembleRetriever,
ParentDocumentRetriever,
ContextualCompressionRetriever,
)

from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain.retrievers.document_compressors import LLMChainExtractor

from langchain.storage import InMemoryStore

from langchain_ollama import ChatOllama, OllamaEmbeddings

# ============================================================

# ENVIRONMENT

# ============================================================

load_dotenv()

# ============================================================

# LOGGING

# ============================================================

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("advanced_rag")

# ============================================================

# CONFIG

# ============================================================

@dataclass
class Settings:
OLLAMA_BASE_URL: str = "[http://localhost:11434](http://localhost:11434)"

```
LLM_MODEL: str = "llama3"
EMBEDDING_MODEL: str = "nomic-embed-text"

CHROMA_DIR: str = "./chroma_db"

RETRIEVAL_K: int = 3

TEMPERATURE: float = 0.0
```

settings = Settings()

# ============================================================

# LOCAL MODELS

# ============================================================

logger.info("Loading Ollama models...")

llm = ChatOllama(
model=settings.LLM_MODEL,
temperature=settings.TEMPERATURE,
base_url=settings.OLLAMA_BASE_URL,
)

embeddings = OllamaEmbeddings(
model=settings.EMBEDDING_MODEL,
base_url=settings.OLLAMA_BASE_URL,
)

logger.info("Models loaded successfully.")

# ============================================================

# SAMPLE DOCUMENTS

# ============================================================

TECH_DOCS = [
Document(
page_content="""
LangChain is a framework for building LLM applications.
It provides tools for prompts, chains, agents, memory,
and retrieval systems.
""",
metadata={
"topic": "ai",
"framework": "langchain",
},
),
Document(
page_content="""
LangGraph is used for stateful AI agents.
It supports cycles, memory, persistence,
human-in-the-loop workflows, and graph orchestration.
""",
metadata={
"topic": "ai",
"framework": "langgraph",
},
),
Document(
page_content="""
Chroma is a vector database used for semantic search.
It stores embeddings and supports similarity retrieval.
""",
metadata={
"topic": "database",
"type": "vector",
},
),
Document(
page_content="""
BM25 is a keyword-based retrieval algorithm.
It is useful for exact term matching.
""",
metadata={
"topic": "retrieval",
"type": "keyword",
},
),
]

# ============================================================

# VECTOR STORE

# ============================================================

def create_vectorstore() -> Chroma:
"""
Create persistent Chroma vector store.
"""

```
logger.info("Creating Chroma vector store...")

vectorstore = Chroma.from_documents(
    documents=TECH_DOCS,
    embedding=embeddings,
    persist_directory=settings.CHROMA_DIR,
    collection_name="advanced_rag_collection",
)

logger.info("Vector store ready.")

return vectorstore
```

# ============================================================

# HYBRID RETRIEVER

# ============================================================

def create_hybrid_retriever(vectorstore: Chroma):
"""
Create BM25 + semantic hybrid retriever.
"""

```
logger.info("Creating hybrid retriever...")

bm25 = BM25Retriever.from_documents(TECH_DOCS)
bm25.k = settings.RETRIEVAL_K

semantic = vectorstore.as_retriever(
    search_kwargs={
        "k": settings.RETRIEVAL_K
    }
)

ensemble = EnsembleRetriever(
    retrievers=[bm25, semantic],
    weights=[0.4, 0.6],
)

logger.info("Hybrid retriever ready.")

return ensemble
```

# ============================================================

# MULTI QUERY RETRIEVER

# ============================================================

def create_multi_query_retriever(base_retriever):
"""
Improve recall using multiple generated queries.
"""

```
logger.info("Creating multi-query retriever...")

retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
)

logger.info("Multi-query retriever ready.")

return retriever
```

# ============================================================

# CONTEXTUAL COMPRESSION

# ============================================================

def create_compression_retriever(base_retriever):
"""
Extract only relevant information.
"""

```
logger.info("Creating compression retriever...")

compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

logger.info("Compression retriever ready.")

return compression_retriever
```

# ============================================================

# PARENT DOCUMENT RETRIEVER

# ============================================================

def create_parent_retriever():
"""
Parent-child retrieval architecture.
"""

```
logger.info("Creating parent retriever...")

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=50,
)

vectorstore = Chroma(
    collection_name=f"parent_docs_{uuid.uuid4().hex}",
    embedding_function=embeddings,
    persist_directory=settings.CHROMA_DIR,
)

store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(TECH_DOCS)

logger.info("Parent retriever ready.")

return retriever
```

# ============================================================

# RAG PROMPT

# ============================================================

PROMPT = ChatPromptTemplate.from_template(
"""
You are a production AI assistant.

Answer ONLY using the provided context.

If the answer is not available in context,
say:
"I do not have enough information."

Context:
{context}

Question:
{question}

Answer:
"""
)

# ============================================================

# FORMATTER

# ============================================================

def format_docs(docs: List[Document]) -> str:
"""
Format retrieved documents.
"""

```
return "\n\n".join(doc.page_content for doc in docs)
```

# ============================================================

# ADVANCED RAG SYSTEM

# ============================================================

class AdvancedRAG:
"""
Production-ready advanced RAG pipeline.
"""

```
def __init__(self):

    logger.info("Initializing AdvancedRAG system...")

    self.vectorstore = create_vectorstore()

    hybrid = create_hybrid_retriever(self.vectorstore)

    multi_query = create_multi_query_retriever(hybrid)

    self.retriever = create_compression_retriever(multi_query)

    self.chain = (
        {
            "context": self.retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    logger.info("AdvancedRAG initialized successfully.")

def ask(self, question: str) -> str:
    """
    Ask question to RAG system.
    """

    logger.info(f"Question received: {question}")

    try:
        response = self.chain.invoke(question)

        logger.info("Response generated successfully.")

        return response

    except Exception as e:
        logger.exception("RAG pipeline failed.")

        return f"Error: {str(e)}"
```

# ============================================================

# DEMO

# ============================================================

def run_demo():

```
rag = AdvancedRAG()

questions = [
    "What is LangChain?",
    "What is LangGraph used for?",
    "What database stores embeddings?",
    "How does BM25 work?",
]

print("\n")
print("=" * 70)
print("ADVANCED RAG DEMO")
print("=" * 70)

for q in questions:

    print("\n")
    print(f"Q: {q}")

    response = rag.ask(q)

    print(f"A: {response}")

    print("-" * 70)
```

# ============================================================

# MAIN

# ============================================================

if **name** == "**main**":

```
run_demo()
```

"""
Production Upgrades To Add Later
================================

1. Redis Cache
2. Async Retrieval
3. Streaming Responses
4. Reranking Models
5. Guardrails
6. LangSmith Tracing
7. API Layer (FastAPI)
8. Authentication
9. Monitoring
10. Rate Limiting
    """
