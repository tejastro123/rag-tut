
"""
Production Monitoring & Logging
Structured logging, metrics, retries, tracing, and observability
"""

import os
import json
import time
import uuid
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from threading import Lock
from typing import Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from langsmith import traceable
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

load_dotenv()

# CONFIG
@dataclass
class Settings:
    ollama_url: str = os.getenv("OLLAMA_URL","http://localhost:11434")
    model: str = os.getenv("MODEL", "mistral")

    temperature: float = 0.0
    request_timeout: int = 30
    max_retries: int = 3

    log_level: str = "INFO"
    log_file: str = "logs/app.log"

    enable_console_logging: bool = True

    def validate(self):
        if not self.ollama_url:
            raise ValueError(
                "OLLAMA_URL missing in environment variables"
            )



# STRUCTURED JSON LOGGER
class JSONFormatter(logging.Formatter):
    """
    Structured JSON logging formatter.
    """

    def format(self, record):

        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "message": record.getMessage(),
        }

        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)

# SETUP logger
def setup_logger(settings: Settings) -> logging.Logger:

    logger = logging.getLogger("production_ai_app")

    if logger.handlers:
        return logger

    logger.setLevel(settings.log_level)

    formatter = JSONFormatter()

    # Console logging
    if settings.enable_console_logging:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File logging with rotation
    os.makedirs("logs", exist_ok=True)

    file_handler = RotatingFileHandler(
        settings.log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# METRICS
class MetricsCollector:
    """
    Thread-safe metrics collector.
    """

    def __init__(self):

        self._lock = Lock()

        self.metrics = {
            "requests_total": 0,
            "errors_total": 0,
            "latency_total_ms": 0.0,
            "latency_count": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def record(
        self,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        error: bool = False,
        cache_hit: bool = False,
    ):

        with self._lock:

            self.metrics["requests_total"] += 1
            self.metrics["latency_total_ms"] += latency_ms
            self.metrics["latency_count"] += 1
            self.metrics["tokens_input"] += input_tokens
            self.metrics["tokens_output"] += output_tokens

            if error:
                self.metrics["errors_total"] += 1

            if cache_hit:
                self.metrics["cache_hits"] += 1
            else:
                self.metrics["cache_misses"] += 1

    def summary(self) -> dict:

        with self._lock:

            total_requests = self.metrics["requests_total"]

            avg_latency = (
                self.metrics["latency_total_ms"]
                / self.metrics["latency_count"]
                if self.metrics["latency_count"] > 0
                else 0
            )

            error_rate = (
                self.metrics["errors_total"] / total_requests
                if total_requests > 0
                else 0
            )

            cache_hit_rate = (
                self.metrics["cache_hits"]
                / (self.metrics["cache_hits"] + self.metrics["cache_misses"])
                if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0
                else 0
            )

            return {
                "requests_total": total_requests,
                "errors_total": self.metrics["errors_total"],
                "error_rate": round(error_rate * 100, 2),
                "avg_latency_ms": round(avg_latency, 2),
                "tokens_input": self.metrics["tokens_input"],
                "tokens_output": self.metrics["tokens_output"],
                "cache_hit_rate": round(cache_hit_rate * 100, 2),
            }


# TOKEN ESTIMATION
def estimate_tokens(text: str) -> int:
    """
    Rough token estimation.
    """
    if not text:
        return 0

    return max(1, int(len(text.split()) * 1.3))


# PRODUCTION LLM SERVICE
class ProductionLLMService:

    def __init__(self, settings: Optional[Settings] = None):

        self.settings = settings or Settings()
        self.settings.validate()

        self.logger = setup_logger(self.settings)
        self.metrics = MetricsCollector()

        self.llm = ChatOllama(
            model=self.settings.model,
            temperature=self.settings.temperature
        )

        self.logger.info(                                                         
            "LLM service initialized",
            extra={
                "extra_data": {
                    "model": self.settings.model,
                }
            },
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    @traceable(name="production_llm_invoke")
    def invoke(self, query: str) -> str:

        request_id = str(uuid.uuid4())

        start_time = time.perf_counter()

        self.logger.info(
            "LLM request started",
            extra={
                "request_id": request_id,
                "extra_data": {
                    "query_preview": query[:100],
                },
            },
        )

        try:

            response = self.llm.invoke(
                [HumanMessage(content=query)]
            )

            result = response.content

            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            input_tokens = estimate_tokens(query)
            output_tokens = estimate_tokens(result)

            self.metrics.record(
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error=False,
                cache_hit=False,
            )

            self.logger.info(
                "LLM request completed",
                extra={
                    "request_id": request_id,
                    "extra_data": {
                        "latency_ms": round(latency_ms, 2),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                },
            )

            return result

        except Exception as e:

            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            self.metrics.record(
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                error=True,
            )

            self.logger.exception(
                "LLM request failed",
                extra={
                    "request_id": request_id,
                    "extra_data": {
                        "latency_ms": round(latency_ms, 2),
                        "error": str(e),
                    },
                },
            )

            raise RuntimeError(
                f"LLM invocation failed: {e}"
            ) from e

    def health_check(self) -> bool:
        """
        Simple health check.
        """

        try:

            response = self.invoke("Reply with: OK")

            return "OK" in response

        except Exception:
            return False

    def print_metrics(self):

        summary = self.metrics.summary()

        print("\n========== METRICS ==========")

        for key, value in summary.items():
            print(f"{key}: {value}")

        print("=============================\n")


# =========================================================
# DEMO
# =========================================================


def main():

    print("=" * 60)
    print("PRODUCTION MONITORING DEMO")
    print("=" * 60)

    service = ProductionLLMService()

    print("\nRunning health check...")

    if service.health_check():
        print("✅ Service healthy")
    else:
        print("❌ Service unhealthy")
        return

    queries = [
        "What is Python?",
        "Explain vector databases.",
        "What is RAG?",
    ]

    for query in queries:
        print(f"\n🔍 Query: {query}")
        try:
            response = service.invoke(query)
            print(f"✅ Response: {response[:100]}...")
        except Exception as e:
            print(f"❌ Failed: {e}")

    service.print_metrics()
    print("\n✅ Production monitoring demo complete")


if __name__ == "__main__":
    main()

