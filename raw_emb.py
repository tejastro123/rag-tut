import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# emb = GoogleGenerativeAIEmbeddings(
#     google_api_key=os.getenv("GOOGLE_API_KEY"),
#     model="gemini-embedding-001"
# )

# embedding = emb.embed_query("Hello, how are you?")

# print("Embedding length:", len(embedding))
# print("First 10 values:", embedding[:10])

from langchain_huggingface import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

embedding = emb.embed_query("Hello, how are you?")

print("Embedding length:", len(embedding))
print("First 10 values:", embedding[:10])
