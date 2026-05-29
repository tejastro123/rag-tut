from app.llm_factory import create_llm
from app.config import settings

class ModelRouter:

def __init__(self):

    self.cheap_model = create_llm(
        settings.CHEAP_MODEL
    )

    self.expensive_model = create_llm(
        settings.EXPENSIVE_MODEL
    )

def classify_complexity(self, query: str):

    complex_keywords = [
        "analyze",
        "compare",
        "optimize",
        "architecture",
        "multi-step",
        "strategy",
        "reasoning",
    ]

    if len(query.split()) > 20:
        return "complex"

    if any(
        keyword in query.lower()
        for keyword in complex_keywords
    ):
        return "complex"

    return "simple"

def invoke(self, query: str):

    complexity = self.classify_complexity(
        query
    )

    if complexity == "simple":
        llm = self.cheap_model
        model_name = settings.CHEAP_MODEL
    else:
        llm = self.expensive_model
        model_name = settings.EXPENSIVE_MODEL

    response = llm.invoke(query)

    return {
        "query": query,
        "model": model_name,
        "complexity": complexity,
        "response": response.content,
    } 
