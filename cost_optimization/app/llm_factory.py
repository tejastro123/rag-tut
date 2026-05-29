from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings

def create_llm(model_name: str):


return ChatOllama(
    model=model_name,
    base_url=settings.OLLAMA_BASE_URL,
    temperature=0.2,
)


def create_embeddings():


return HuggingFaceEmbeddings(
    model_name=settings.EMBEDDING_MODEL
)

