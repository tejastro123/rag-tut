"""
Production PGVector Connection (Neon / PostgreSQL)

Features:
- Neon PostgreSQL support
- pgvector integration
- LangChain PGVector
- Semantic search
- Health checks
- Production logging
"""

import os
import logging
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# -------------------------------------------------------------------
# Load Environment Variables
# -------------------------------------------------------------------

load_dotenv()

# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Environment Configuration
# -------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError(
        "\n❌ DATABASE_URL missing in .env\n\n"
        "Example:\n"
        "DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require\n"
    )

# Required for langchain-postgres + psycopg3
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgresql://",
        "postgresql+psycopg://",
        1,
    )

COLLECTION_NAME = "production_docs"

# -------------------------------------------------------------------
# Embedding Model
# -------------------------------------------------------------------

logger.info("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={
        "device": "cpu",
    },
    encode_kwargs={
        "normalize_embeddings": True,
    },
)

logger.info("✅ Embedding model loaded")

# -------------------------------------------------------------------
# Database Health Check
# -------------------------------------------------------------------


def verify_database_connection() -> bool:
    """
    Verify raw PostgreSQL connectivity.
    """

    try:
        logger.info("Checking PostgreSQL connection...")

        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_recycle=300,
        )

        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        logger.info("✅ PostgreSQL connection successful")

        return True

    except SQLAlchemyError as e:
        logger.exception("❌ PostgreSQL connection failed")
        raise RuntimeError(f"Database connection failed: {e}")


# -------------------------------------------------------------------
# Connect PGVector
# -------------------------------------------------------------------


def connect_vectorstore() -> PGVector:
    """
    Connect to PGVector vector store.
    """

    try:
        logger.info("Initializing PGVector...")

        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=DATABASE_URL,
            use_jsonb=True,
            create_extension=False,
            pre_delete_collection=False,
        )

        logger.info("✅ PGVector initialized")

        return vectorstore

    except Exception as e:
        logger.exception("❌ PGVector initialization failed")
        raise RuntimeError(f"PGVector initialization failed: {e}")


# -------------------------------------------------------------------
# Vector Store Health Check
# -------------------------------------------------------------------


def verify_vectorstore(vectorstore: PGVector) -> bool:
    """
    Verify vector operations.
    """

    test_doc = Document(
        page_content="Vector database health check document.",
        metadata={
            "type": "healthcheck",
            "environment": "production",
        },
    )

    inserted_ids = []

    try:
        logger.info("Running vector store health check...")

        # Insert
        inserted_ids = vectorstore.add_documents([test_doc])

        logger.info(f"✅ Insert successful: {inserted_ids[0]}")

        # Search
        results = vectorstore.similarity_search(
            "health check",
            k=1,
        )

        if results:
            logger.info("✅ Similarity search successful")

        # Cleanup
        vectorstore.delete(inserted_ids)

        logger.info("✅ Cleanup successful")

        return True

    except Exception as e:
        logger.exception("❌ Vector store verification failed")

        if inserted_ids:
            try:
                vectorstore.delete(inserted_ids)
            except Exception:
                pass

        return False


# -------------------------------------------------------------------
# Add Documents
# -------------------------------------------------------------------


def add_documents(
    vectorstore: PGVector,
    documents: list[str],
    metadata: Optional[list[dict]] = None,
):
    """
    Add documents to vector database.
    """

    try:
        docs = []

        for index, text_content in enumerate(documents):

            doc_metadata = metadata[index] if metadata else {}

            docs.append(
                Document(
                    page_content=text_content,
                    metadata=doc_metadata,
                )
            )

        ids = vectorstore.add_documents(docs)

        logger.info(f"✅ Added {len(ids)} documents")

        return ids

    except Exception as e:
        logger.exception("❌ Document insertion failed")
        raise RuntimeError(f"Document insertion failed: {e}")


# -------------------------------------------------------------------
# Search Documents
# -------------------------------------------------------------------


def search_documents(
    vectorstore: PGVector,
    query: str,
    k: int = 3,
):
    """
    Perform semantic similarity search.
    """

    try:
        results = vectorstore.similarity_search(
            query=query,
            k=k,
        )

        logger.info(f"✅ Retrieved {len(results)} results")

        return results

    except Exception as e:
        logger.exception("❌ Similarity search failed")
        raise RuntimeError(f"Search failed: {e}")


# -------------------------------------------------------------------
# Main Application
# -------------------------------------------------------------------


def main():

    print("=" * 70)
    print("PGVECTOR PRODUCTION CONNECTION")
    print("=" * 70)

    try:
        # Step 1: Verify PostgreSQL
        verify_database_connection()

        # Step 2: Initialize PGVector
        vectorstore = connect_vectorstore()

        # Step 3: Verify Vector Operations
        success = verify_vectorstore(vectorstore)

        if not success:
            print("\n❌ Vector store health check failed")
            return

        print("\n✅ PGVector ready for production")

        # -------------------------------------------------------------------
        # Sample Data
        # -------------------------------------------------------------------

        sample_docs = [
            "LangChain is a framework for building LLM applications.",
            "Neon provides serverless PostgreSQL with pgvector support.",
            "RAG combines retrieval with large language models.",
        ]

        sample_metadata = [
            {"topic": "langchain"},
            {"topic": "database"},
            {"topic": "rag"},
        ]

        # Insert Documents
        add_documents(
            vectorstore=vectorstore,
            documents=sample_docs,
            metadata=sample_metadata,
        )

        # -------------------------------------------------------------------
        # Semantic Search Demo
        # -------------------------------------------------------------------

        print("\n🔍 Running semantic search...\n")

        results = search_documents(
            vectorstore=vectorstore,
            query="What is vector search?",
            k=2,
        )

        for index, document in enumerate(results, start=1):

            print(f"{index}. {document.page_content}")
            print(f"   Metadata: {document.metadata}\n")

    except Exception as e:
        logger.exception("❌ Application crashed")
        print(f"\n❌ Fatal Error: {e}")


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()