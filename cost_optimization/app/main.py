from app.router import ModelRouter
from app.cache import PersistentCache
from app.rag import LocalRAG
from app.budget import TokenBudget

def demo_router():

print("\\n=== MODEL ROUTING ===\\n")

router = ModelRouter()

queries = [
    "What is Python?",
    "Analyze AI impact on jobs.",
]

for query in queries:

    result = router.invoke(query)

    print(f"Query: {query}")
    print(f"Model: {result['model']}")
    print(
        f"Complexity: {result['complexity']}"
    )

    print(
        f"Response: {result['response']}"
    )

    print("-" * 50)

def demo_cache():

print("\\n=== CACHE DEMO ===\\n")

cache = PersistentCache()

query = "What is Python?"

cached = cache.get(query)

if cached:

    print("CACHE HIT")
    print(cached)

else:

    print("CACHE MISS")

    response = (
        "Python is a programming language."
    )

    cache.set(query, response)

    print("Stored in cache.")


def demo_budget():

print("\\n=== TOKEN BUDGET ===\\n")

budget = TokenBudget(max_tokens=100)

text = (
    "Explain artificial intelligence."
)

tokens = budget.validate(text)

print(f"Token count: {tokens}")


def demo_rag():

print("\\n=== LOCAL RAG ===\\n")

knowledge = '''

Python is a programming language.

It was created by Guido van Rossum.

Python 3.12 improved error messages.
'''

rag = LocalRAG()

rag.build_knowledge_base(
    knowledge,
    source="python_docs",
)

chain = rag.create_chain()

question = (
    "Who created Python?"
)

answer = chain.invoke(question)

print(f"Question: {question}")
print(f"Answer: {answer}")


if **name** == "**main**":


demo_router()

demo_cache()

demo_budget()

demo_rag()

