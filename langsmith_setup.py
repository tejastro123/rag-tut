"""
LangSmith Setup and Observability
Production monitoring for LangChain/LangGraph
"""

import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from langsmith.run_trees import RunTree
from dotenv import load_dotenv

load_dotenv()

# Enable tracing
os.environ["LANGSMITH_TRACING"] = "true"

# Optional project name
os.environ["LANGSMITH_PROJECT"] = "multi-agent-researcher"

# Ollama base URL
OLLAMA_BASE_URL = "http://localhost:11434"

@traceable(name="basic_chaining")
def demo_basic_tracing():
    """Basic LangSmith tracing."""

    llm = ChatOllama(
        model="llama3",
        temperature=0,
        base_url=OLLAMA_BASE_URL,
    )

    prompt = ChatPromptTemplate.from_template("Explain {topic} in one sentence.")

    chain = prompt | llm | StrOutputParser()

    print("Basic Tracing Demo:\n")
    print("Running chain with LangSmith tracing enabled...")

    result = chain.invoke({"topic": "machine learning"})

    print(f"Result: {result}")
    print("\nCheck LangSmith dashboard for trace details.")


@traceable(name="named_runs_demo", tags=["production", "summarization"])
def demo_named_runs():
    """Name your runs for easier identification."""

    llm = ChatOllama(
        model="llama3",
        temperature=0,
        base_url=OLLAMA_BASE_URL,
    )

    prompt = ChatPromptTemplate.from_template("Summarize: {text}")

    chain = prompt | llm | StrOutputParser()

    print("\nNamed Runs Demo:\n")

    result = chain.invoke(
        {"text": "LangSmith provides observability for LLM applications."}
    )

    print(f"Result: {result}")
    print("Run tagged with 'production', 'summarization'")


@traceable(name="trace_with_metadata_demo", tags=["metadata", "filtering"])
def demo_trace_with_metadata(user_id: str, request_type: str):
    """Add metadata to traces for filtering."""

    llm = ChatOllama(
        model="llama3",
        temperature=0,
        base_url=OLLAMA_BASE_URL,
    )

   # Metadata automatically appears in LangSmith traces
    result = llm.invoke(
        f"Hello from user {user_id}. "
        f"Request type: {request_type}"
    )

    return result.content


def demo_manual_run_tree():
    """Advanced LangSmith manual tracing."""

    print("\nManual RunTree Demo:\n")

    run_tree = RunTree(
        name="manual_pipeline",
        run_type="chain",
        inputs={
            "topic": "LangSmith observability"
        },
    )

    run_tree.post()

    try:
        llm = ChatOllama(
            model="llama3",
            temperature=0,
            base_url=OLLAMA_BASE_URL,
        )

        response = llm.invoke(
            "Explain LangSmith observability in one sentence."
        )

        run_tree.end(
            outputs={
                "response": response.content
            }
        )

        print(response.content)

    except Exception as e:
        run_tree.end(
            error=str(e)
        )
        raise

    finally:
        run_tree.patch()


if __name__ == "__main__":
    demo_basic_tracing()
    demo_named_runs()

    result = demo_trace_with_metadata(
        user_id="user_123",
        request_type="greeting",
    )

    print(f"\nMetadata Trace Result:\n{result}")

    demo_manual_run_tree()
