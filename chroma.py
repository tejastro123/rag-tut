import chromadb
chroma_client = chromadb.Client()

collection_name = "test_collection"
collection = chroma_client.get_or_create_collection(name=collection_name)

# define text documents
documents = [
    {"id": "doc1", "text": "Hello, world!"},
    {"id": "doc2", "text": "How are you today?"},
    {"id": "doc3", "text": "Goodbye, see you later!"},
    {
        "id": "doc4",
        "text": "Microsoft is a technology company that develops software. It was founded by Bill Gates and Paul Allen in 1975.",
    },
]

# add documents to collection
# collection.add(
#     ids=["id1", "id2"],
#     documents=[
#         "This is a document about pineapple",
#         "This is a document about oranges"
#     ]
# )
for doc in documents:
    collection.upsert(ids=[doc["id"]], documents=[doc["text"]])

print("--- collection.get() ---\n",collection.get())

# define query
query = "What is Microsoft?"

results = collection.query(
    query_texts=[query], # Chroma will embed this for you
    n_results=2 # how many results to return
)
print("--- results ---\n", results)

