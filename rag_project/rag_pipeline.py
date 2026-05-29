from typing import List
import logging

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    EMBEDDING_MODEL,
    CHROMA_DB_DIR,
    TOP_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        logger.info("Initializing embeddings model...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        logger.info("Initializing Ollama model...")

        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,
        )

        self.prompt = ChatPromptTemplate.from_template(
            """
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is not available in the context,
say: "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
        )

        self.output_parser = StrOutputParser()

        self.vectorstore = None
        self.retriever = None
        self.chain = None

    def ingest_documents(self, texts: List[str]):
        logger.info("Splitting documents...")

        docs = [Document(page_content=t) for t in texts]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        split_docs = splitter.split_documents(docs)

        logger.info(f"Created {len(split_docs)} chunks")

        logger.info("Creating Chroma vector store...")

        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=CHROMA_DB_DIR,
        )

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K},
        )

        self.chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | self.output_parser
        )

        logger.info("RAG pipeline ready.")

    def ask(self, question: str) -> str:
        if not self.chain:
            raise ValueError("Pipeline not initialized. Call ingest_documents() first.")

        logger.info(f"Question: {question}")

        try:
            response = self.chain.invoke(question)
            return response

        except Exception as e:
            logger.error(f"RAG Error: {e}")
            return "An error occurred while processing your request."

    @staticmethod
    def _format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
