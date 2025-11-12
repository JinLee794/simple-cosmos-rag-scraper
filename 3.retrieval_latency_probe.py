#!/usr/bin/env python3
"""Benchmark Cosmos DB vector retrieval latency for a set of queries."""

from __future__ import annotations

import argparse
import math
import sys
import textwrap
import time
from typing import Iterable, List, Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from src.cosmos_vector_retriever import (
    CosmosVectorRetriever,
    DEFAULT_NUM_CANDIDATES,
    DEFAULT_TOP_K,
    RetrievalResult,
)


def configure_tracing(enable_tracing: bool) -> None:
    if not enable_tracing:
        return
    provider = TracerProvider()
    processor = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        dest="queries",
        action="append",
        help="Query text to benchmark (repeatable). If omitted, use --query-file",
    )
    parser.add_argument(
        "--query-file",
        help="Path to a text file with one query per line.",
    )
    parser.add_argument("--iterations", type=int, default=5, help="Number of timed iterations per query.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per query (not timed).")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--num-candidates", type=int, default=DEFAULT_NUM_CANDIDATES)
    parser.add_argument("--vector-field", default="embedding")
    parser.add_argument("--similarity", default="cosine")
    parser.add_argument("--enable-tracing", action="store_true", help="Emit OpenTelemetry spans to stdout.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def load_queries(args: argparse.Namespace) -> List[str]:
    queries: List[str] = []
    if args.queries:
        queries.extend(q.strip() for q in args.queries if q and q.strip())
    if args.query_file:
        with open(args.query_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    queries.append(line)
    if not queries:
        raise ValueError("Provide queries via --query or --query-file.")
    return queries


def summarize_results(durations: List[float]) -> str:
    durations_ms = [d * 1000.0 for d in durations]
    count = len(durations_ms)
    avg = sum(durations_ms) / count
    sorted_ms = sorted(durations_ms)
    index = min(count - 1, max(0, math.ceil(0.95 * count) - 1))
    p95 = sorted_ms[index]
    return (
        f"count={count} min={sorted_ms[0]:.2f}ms "
        f"avg={avg:.2f}ms max={sorted_ms[-1]:.2f}ms p95={p95:.2f}ms"
    )


def format_results(results: List[RetrievalResult], limit: int = 1) -> str:
    lines: List[str] = []
    for result in results[:limit]:
        snippet = textwrap.shorten(" ".join(result.content.split()), width=160, placeholder="…")
        score = result.score if result.score is None else f"{result.score:.4f}"
        lines.append(f"- url={result.url} score={score} snippet={snippet}")
    return "\n".join(lines)


def benchmark_query(
    retriever: CosmosVectorRetriever,
    query: str,
    iterations: int,
    warmup: int,
    top_k: int,
    num_candidates: int,
) -> List[float]:
    for _ in range(max(0, warmup)):
        retriever.search(query, top_k=top_k, num_candidates=num_candidates)
    durations: List[float] = []
    for _ in range(max(1, iterations)):
        start = time.perf_counter()
        retriever.search(query, top_k=top_k, num_candidates=num_candidates)
        durations.append(time.perf_counter() - start)
    return durations


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    configure_tracing(args.enable_tracing)

    try:
        queries = load_queries(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    retriever = CosmosVectorRetriever.from_env(
        vector_field=args.vector_field,
        similarity=args.similarity,
        appname="venmo-latency-probe",
    )

    print(
        f"Running latency probe for {len(queries)} queries with iterations={args.iterations} warmup={args.warmup}"
    )

    for query in queries:
        print("\nQuery:", query)
        durations = benchmark_query(
            retriever,
            query=query,
            iterations=args.iterations,
            warmup=args.warmup,
            top_k=args.top_k,
            num_candidates=args.num_candidates,
        )
        summary = summarize_results(durations)
        print("Latency:", summary)
        if args.verbose:
            results = retriever.search(query, top_k=args.top_k, num_candidates=args.num_candidates)
            if results:
                print("Top results:")
                print(format_results(results))
            else:
                print("No results returned.")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
