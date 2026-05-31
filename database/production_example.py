
"""
Production-Ready RAG Service
Neon + PGVector + LangChain + OpenAI

Features:
- Production logging
- Neon PostgreSQL support
- pgvector vector search
- RAG pipeline
- Health checks
- Metadata filtering
- Source attribution
- Connection pooling support
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ollama
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

# Environment
load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# Configuration
@dataclass
class Config:

    # Database
    database_url: str = os.getenv("DATABASE_URL", "")

    # Collection
    collection_name: str = "production_documents"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # LLM
    chat_model: str = "mistral"
    temperature: float = 0.0

    # Retrieval
    default_k: int = 5

    # Search threshold
    min_similarity: float = 0.5

    def __post_init__(self):

        if not self.database_url:
            raise ValueError(
                "\n❌ DATABASE_URL missing in .env\n"
            )

        # Required for langchain-postgres
        if self.database_url.startswith("postgresql://"):
            self.database_url = self.database_url.replace(
                "postgresql://",
                "postgresql+psycopg://",
                1,
            )


# RAG Service
class RAGService:
    """
    Production-ready RAG service.
    """

    def __init__(self, config: Optional[Config] = None):

        self.config = config or Config()

        self._vectorstore = None
        self._chain = None
        self._embeddings = None
        self._llm = None

        self._verify_database_connection()

    # Database Health Check
    def _verify_database_connection(self):

        try:

            logger.info("Checking PostgreSQL connection...")

            engine = create_engine(
                self.config.database_url,
                pool_pre_ping=True,
                pool_recycle=300,
            )

            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            logger.info("✅ PostgreSQL connection successful")

        except SQLAlchemyError as e:

            logger.exception("❌ Database connection failed")

            raise RuntimeError(
                f"Database connection failed: {e}"
            )

    # Embeddings
    @property
    def embeddings(self):

        if self._embeddings is None:

            logger.info("Loading embedding model...")

            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model
            )

            logger.info("✅ Embedding model loaded")

        return self._embeddings

    # LLM
    @property
    def llm(self):

        if self._llm is None:

            self._llm = ChatOllama(
                model=self.config.chat_model,
                temperature=self.config.temperature,
            )

        return self._llm

    # Vector Store
    @property
    def vectorstore(self) -> PGVector:

        if self._vectorstore is None:

            logger.info("Initializing PGVector...")

            self._vectorstore = PGVector(
                embeddings=self.embeddings,
                collection_name=self.config.collection_name,
                connection=self.config.database_url,
                use_jsonb=True,
                create_extension=False,
                pre_delete_collection=False,
            )

            logger.info("✅ PGVector initialized")

        return self._vectorstore

    # Create RAG Chain
    @property
    def chain(self):

        if self._chain is None:
            self._chain = self._create_chain()

        return self._chain

    def _create_chain(self):

        retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": self.config.default_k,
            }
        )

        prompt = ChatPromptTemplate.from_template(
            """
You are a professional AI assistant.

Answer the user's question ONLY using the provided context.

If the answer is not available in the context,
say:

"I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:
"""
        )

        def format_docs(docs):

            return "\n\n".join(
                doc.page_content
                for doc in docs
            )

        chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    # Add Documents
    def add_documents(
        self,
        documents: list[Document],
    ) -> list[str]:

        try:

            ids = self.vectorstore.add_documents(
                documents
            )

            logger.info(
                f"✅ Added {len(ids)} documents"
            )

            return ids

        except Exception as e:

            logger.exception(
                "❌ Failed to add documents"
            )

            raise RuntimeError(
                f"Document insertion failed: {e}"
            )

    # Searchs
    def search(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[dict] = None,
    ):

        try:

            search_kwargs = {
                "k": k or self.config.default_k,
            }

            if filter_dict:
                search_kwargs["filter"] = filter_dict

            results = (
                self.vectorstore
                .similarity_search_with_score(
                    query=query,
                    **search_kwargs,
                )
            )

            logger.info(
                f"✅ Retrieved {len(results)} documents"
            )

            return results

        except Exception as e:

            logger.exception(
                "❌ Search failed"
            )

            raise RuntimeError(
                f"Search failed: {e}"
            )

    # Ask Question
    def ask(self, question: str) -> str:

        try:

            response = self.chain.invoke(question)

            return response

        except Exception as e:

            logger.exception(
                "❌ RAG generation failed"
            )

            raise RuntimeError(
                f"Generation failed: {e}"
            )

    # Ask with Sources
    def ask_with_sources(
        self,
        question: str,
    ) -> dict:

        docs_with_scores = self.search(question)

        answer = self.ask(question)

        formatted_sources = []

        for doc, score in docs_with_scores:

            formatted_sources.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score),
                }
            )

        return {
            "question": question,
            "answer": answer,
            "sources": formatted_sources,
        }


# Demo
def main():

    print("=" * 70)
    print("PRODUCTION RAG SERVICE")
    print("=" * 70)

    try:

        # Initialize
        logger.info("Initializing RAG service...")

        service = RAGService()

        logger.info("✅ RAG service ready")

        # -------------------------------------------------------------------
        # Sample Documents
        # -------------------------------------------------------------------

        documents = [
            Document(
                page_content=(
                    "pgvector is a PostgreSQL extension "
                    "for vector similarity search."
                ),
                metadata={
                    "topic": "pgvector",
                    "source": "docs",
                },
            ),
            Document(
                page_content=(
                    "HNSW indexes improve approximate "
                    "nearest neighbor search performance."
                ),
                metadata={
                    "topic": "indexing",
                    "source": "docs",
                },
            ),
            Document(
                page_content=(
                    "Connection pooling is essential "
                    "for production PostgreSQL systems."
                ),
                metadata={
                    "topic": "production",
                    "source": "best-practices",
                },
            ),
        ]

        logger.info("Adding sample documents...")

        ids = service.add_documents(documents)

        logger.info(
            f"✅ Added {len(ids)} documents"
        )

        #Search Demo
        print("\n🔍 Semantic Search Demo\n")

        search_results = service.search(
            query="How can I optimize vector search?",
            k=2,
        )

        for index, (doc, score) in enumerate(
            search_results,
            start=1,
        ):

            print(f"{index}. Score: {score:.4f}")
            print(f"   Content: {doc.page_content}")
            print(f"   Metadata: {doc.metadata}\n")

        # RAG Demo
        question = (
            "What is pgvector and "
            "how can I improve search performance?"
        )

        print("\n❓ RAG Question\n")
        print(question)

        response = service.ask_with_sources(
            question
        )

        print("\n🤖 Answer\n")
        print(response["answer"])

        print("\n📚 Sources\n")

        for index, source in enumerate(
            response["sources"],
            start=1,
        ):

            print(f"{index}. {source['content']}")
            print(
                f"   Similarity: "
                f"{source['similarity_score']:.4f}"
            )
            print(
                f"   Metadata: "
                f"{source['metadata']}\n"
            )

        print("=" * 70)
        print("✅ Production RAG demo complete")
        print("=" * 70)

    except Exception as e:

        logger.exception(
            "❌ Application crashed"
        )

        print(f"\n❌ Fatal Error: {e}")


if __name__ == "__main__":
    main()
