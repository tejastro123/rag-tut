
"""
Production Monitoring & Structured Logging
Production-ready observability for LangChain apps
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict

from dotenv import load_dotenv
from langsmith import traceable
from langchain_ollama import ChatOllama


load_dotenv()

# STRUCTURED JSON LOGGING
class JSONFormatter(logging.Formatter):
    """Format logs as structured JSON."""

    def format(self, record):
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }

        # Add custom fields if available
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        return json.dumps(log_data)


def setup_logging() -> logging.Logger:
    """Configure structured logger."""

    logger = logging.getLogger("production_rag")

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    logger.addHandler(handler)

    return logger


# METRICS COLLECTION
class MetricsCollector:
    """Simple in-memory metrics collector."""

    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "errors_total": 0,
            "latency_sum_ms": 0,
            "latency_count": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def record_request(
        self,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        error: bool = False,
        cache_hit: bool = False,
    ):
        self.metrics["requests_total"] += 1
        self.metrics["latency_sum_ms"] += latency_ms
        self.metrics["latency_count"] += 1
        self.metrics["tokens_input"] += input_tokens
        self.metrics["tokens_output"] += output_tokens

        if error:
            self.metrics["errors_total"] += 1

        if cache_hit:
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1

    def get_summary(self) -> Dict:
        """Return aggregated metrics."""

        avg_latency = (
            self.metrics["latency_sum_ms"]
            / self.metrics["latency_count"]
            if self.metrics["latency_count"] > 0
            else 0
        )

        error_rate = (
            self.metrics["errors_total"]
            / self.metrics["requests_total"]
            if self.metrics["requests_total"] > 0
            else 0
        )

        total_cache = (
            self.metrics["cache_hits"] + self.metrics["cache_misses"]
        )

        cache_hit_rate = (
            self.metrics["cache_hits"] / total_cache
            if total_cache > 0
            else 0
        )

        return {
            "total_requests": self.metrics["requests_total"],
            "total_errors": self.metrics["errors_total"],
            "error_rate": f"{error_rate:.2%}",
            "avg_latency_ms": round(avg_latency, 2),
            "total_input_tokens": self.metrics["tokens_input"],
            "total_output_tokens": self.metrics["tokens_output"],
            "cache_hit_rate": f"{cache_hit_rate:.2%}",
        }


# INSTRUMENTED LLM
class InstrumentedLLM:
    """LLM wrapper with monitoring + structured logging."""

    def __init__(self):
        self.logger = setup_logging()
        self.metrics = MetricsCollector()

        self.llm = ChatOllama(
            model="mistral",
            temperature=0,
        )

    @traceable(name="instrumented_llm_invoke")
    def invoke(self, query: str) -> str:
        """Invoke LLM with monitoring."""

        start_time = time.time()

        try:
            response = self.llm.invoke(query)

            result = response.content

            # Token usage from OpenAI response metadata
            usage = response.response_metadata.get("token_usage", {})

            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            latency_ms = (time.time() - start_time) * 1000

            self.metrics.record_request(
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error=False,
                cache_hit=False,
            )

            self.logger.info(
                "LLM request completed",
                extra={
                    "extra_data": {
                        "latency_ms": round(latency_ms, 2),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
                },
            )

            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            self.metrics.record_request(
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                error=True,
                cache_hit=False,
            )

            self.logger.error(
                "LLM request failed",
                extra={
                    "extra_data": {
                        "error": str(e),
                        "latency_ms": round(latency_ms, 2),
                    }
                },
            )

            raise RuntimeError(f"LLM invocation failed: {e}") from e


# DEMO
def demo_monitoring():
    """Demonstrate monitoring system."""

    print("=" * 60)
    print("PRODUCTION MONITORING DEMO")
    print("=" * 60)

    llm = InstrumentedLLM()

    queries = [
        "What is Python?",
        "Explain machine learning briefly.",
        "What is vector search?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")

        try:
            result = llm.invoke(query)

            print(f"Response: {result[:120]}...")

        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)

    summary = llm.metrics.get_summary()

    for key, value in summary.items():
        print(f"{key}: {value}")


# ENTRY POINT
if __name__ == "__main__":
    demo_monitoring()
