from dotenv import load_dotenv
import os

load_dotenv()

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chroma
CHROMA_DB_DIR = "./data/chroma_db"

# Retrieval
TOP_K = 3

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
