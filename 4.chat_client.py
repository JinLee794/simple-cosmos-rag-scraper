#!/usr/bin/env python3
"""Simple CLI chat client for querying the Cosmos DB vector store."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap
from typing import Iterable, List, Optional

from dotenv import load_dotenv

from src.cosmos_vector_retriever import (
    CosmosVectorRetriever,
    DEFAULT_NUM_CANDIDATES,
    DEFAULT_TOP_K,
    RetrievalResult,
    build_rbac_connection_string,
)

load_dotenv()

DEFAULT_SIMILARITY = "cosine"


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="User question to search against the vector store")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--num-candidates", type=int, default=DEFAULT_NUM_CANDIDATES)
    parser.add_argument("--vector-field", default="embedding")
    parser.add_argument(
        "--database",
        default=os.environ.get("AZURE_COSMOS_DATABASE_NAME", os.environ.get("COSMOS_DATABASE", "venmo")),
    )
    parser.add_argument(
        "--collection",
        default=os.environ.get(
            "AZURE_COSMOS_COLLECTION_NAME",
            os.environ.get("COSMOS_COLLECTION", "help_docs"),
        ),
    )
    parser.add_argument(
        "--connection-string",
        default=os.environ.get("AZURE_COSMOS_CONNECTION_STRING", os.environ.get("COSMOS_CONNECTION_STRING")),
    )
    parser.add_argument("--similarity", default=DEFAULT_SIMILARITY)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def ensure_connection_string(args: argparse.Namespace) -> None:
    if not args.connection_string:
        endpoint = os.environ.get("COSMOS_DB_ENDPOINT")
        if endpoint:
            args.connection_string = build_rbac_connection_string(endpoint)
    if not args.connection_string:
        raise ValueError(
            "Provide a connection string via --connection-string, AZURE_COSMOS_CONNECTION_STRING, or COSMOS_DB_ENDPOINT"
        )


def prompt_for_query() -> str:
    try:
        return input("Enter your question: ").strip()
    except EOFError:
        return ""


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    ensure_connection_string(args)

    query = args.query.strip() if args.query else prompt_for_query()
    if not query:
        logging.error("No query provided")
        return 1

    retriever = CosmosVectorRetriever.from_env(
        connection_string=args.connection_string,
        database=args.database,
        collection=args.collection,
        vector_field=args.vector_field,
        similarity=args.similarity,
        appname="venmo-chat-client",
    )

    results: List[RetrievalResult] = retriever.search(query, top_k=args.top_k, num_candidates=args.num_candidates)

    if not results:
        print("No matches found.")
        return 0

    for idx, item in enumerate(results, start=1):
        raw_doc = item.raw or {}
        score = item.score
        if score is None:
            score = raw_doc.get("$vectorSearchScore") or raw_doc.get("$searchScore")
        url = item.url or raw_doc.get("url") or "(no url)"
        doc_type = item.doc_type or raw_doc.get("doc_type")
        raw_content = item.content or raw_doc.get("content", "")
        snippet = textwrap.shorten(" ".join(raw_content.split()), width=320, placeholder="…")
        print()
        print(f"Result {idx}")
        if score is not None:
            print(f"  Score: {score:.4f}")
        if doc_type:
            print(f"  Doc Type: {doc_type}")
        print(f"  URL: {url}")
        print(f"  Snippet: {snippet}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
