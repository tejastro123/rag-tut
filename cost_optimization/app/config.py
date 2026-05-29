from pydantic_settings import BaseSettings

class Settings(BaseSettings):

OLLAMA_BASE_URL: str = "http://localhost:11434"

CHEAP_MODEL: str = "llama3"
EXPENSIVE_MODEL: str = "llama3"

EMBEDDING_MODEL: str = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

MAX_TOKENS_PER_REQUEST: int = 4000

class Config:
    env_file = ".env"

settings = Settings()
