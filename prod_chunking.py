from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

load_dotenv()

# Local Embedding Model
emb = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 32
        }
)

# Sample document with distinct topics
document = """
# Authentication Guide

## OAuth2 Authentication
To authenticate with our API, you need OAuth2 credentials.
First, obtain a client_id and client_secret from the developer portal.
Make a POST request to /oauth/token with grant_type=client_credentials.
The response contains an access_token valid for 3600 seconds.
Include this token in the Authorization header as 'Bearer <token>'.

## Rate Limiting
Our API implements rate limiting using a token bucket algorithm.
Free tier: 100 requests per minute.
Pro tier: 1000 requests per minute.
Enterprise tier: Custom limits.
When rate limited, you receive a 429 status code.
The Retry-After header indicates when to retry.

## Error Handling
All errors return a standard JSON format.
The 'code' field contains a machine-readable error code.
The 'message' field contains a human-readable description.
Common errors: AUTH_FAILED, RATE_LIMITED, INVALID_REQUEST.
Always check the HTTP status code first, then parse the error body.

## Webhooks
Configure webhooks in your dashboard settings.
We support HTTP and HTTPS endpoints.
Webhook payloads are signed with HMAC-SHA256.
Verify signatures using your webhook secret.
Failed deliveries are retried with exponential backoff.
"""

# Smart chunking
def smart_chunker(
    text: str,
    use_semantic: bool = True,
    fallback_chunk_size: int = 500
) -> list[str]:
    '''
    Production chunking with semantic as primary, recursive as fallback.
    '''
    embeddings = emb
    # Semantic chunking
    if use_semantic:
        try:
            chunker = SemanticChunker(
                embeddings,
                breakpoint_threshold_type='percentile',
                breakpoint_threshold_amount=90
            )
            chunks = chunker.split_text(text)

            # Validate chunks aren't too large
            max_chunk_size = 2000
            if any(len(c) > max_chunk_size for c in chunks):
                # Fallback to recursive for oversized chunks
                print("Oversized semantic chunks detected. Using recursive fallback.")
                return _recursive_fallback(text, fallback_chunk_size)

            return chunks

        except Exception as e:
            print(f'Oversized semantic chunks detected. Semantic chunking failed: {e}, using fallback')
            return _recursive_fallback(text, fallback_chunk_size)

    return _recursive_fallback(text, fallback_chunk_size)

# Recursive fallback
def _recursive_fallback(text: str, chunk_size: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=50
    )
    return splitter.split_text(text)

# Create Chunks
chunks = smart_chunker(document, use_semantic=True)
print(f"\nCreated {len(chunks)} chunks\n")

for i, chunk in enumerate(chunks, start=1):
    print("=" * 60)
    print(f"Chunk {i}")
    print("=" * 60)
    print(chunk)

# ==============================
# Create Vector Store
# ==============================

vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=emb,
    collection_name="auth_docs",
)

# ==============================
# Retrieval Test
# ==============================

queries = [
    "How does OAuth2 authentication work?",
    "What happens when rate limit is exceeded?",
    "How are webhooks secured?",
]

print("\n\nRETRIEVAL TESTS\n")

for query in queries:
    docs = vectorstore.similarity_search(query, k=1)

    print("=" * 60)
    print(f"Query: {query}")
    print("=" * 60)
    print(docs[0].page_content[:300])
    print()