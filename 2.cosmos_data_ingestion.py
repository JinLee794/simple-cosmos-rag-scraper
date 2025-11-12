#!/usr/bin/env python3
"""Ingest Venmo help docs into Azure Cosmos DB for MongoDB vCore with vector search."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse

import pymongo
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
from pymongo.auth_oidc import (
	OIDCCallback,
	OIDCCallbackContext,
	OIDCCallbackResult,
)
from pymongo.collection import Collection
from pymongo.errors import OperationFailure

from dotenv import load_dotenv

load_dotenv()

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_SIMILARITY = "cosine"
DEFAULT_INDEX_KIND = "vector-hnsw"
COSMOS_VECTOR_DIMENSION_LIMIT = 2000

BOILERPLATE_LINES = {
	"You need to enable JavaScript to run this app.",
	"Log In",
	"Help Center",
	"How can we help you?",
	"Venmo",
	"Topics",
	"Resources",
	"Why Venmo",
	"Trust & safety",
	"Money Talks",
	"Our fees",
	"Developers",
	"Company",
	"About us",
	"Jobs",
	"Accessibility",
	"News & Press",
	"Blog",
	"Legal",
	"Terms",
	"Privacy",
	"Cookies",
	"Contact us",
}

BOILERPLATE_PREFIXES = (
	"Send & Receive",
	"Pay with Venmo",
	"Venmo Debit Card",
	"Venmo Credit Card",
	"Venmo for Business",
	"Accept Venmo",
	"Help Center",
)

BOILERPLATE_SUFFIXES = (
	"home_page",
	"section_page",
)

FOOTER_PATTERNS = (
	re.compile(r"^Venmo is a service of PayPal", re.IGNORECASE),
	re.compile(r"©\s?\d{4}\s?PayPal", re.IGNORECASE),
)


def configure_logging(verbose: bool) -> None:
	"""Configure logging with optional verbosity."""

	level = logging.DEBUG if verbose else logging.INFO
	logging.basicConfig(
		level=level,
		format="%(asctime)s %(levelname)s %(name)s - %(message)s",
	)


def load_documents(json_path: Path) -> List[dict]:
	"""Load JSON documents from disk."""

	logging.info("Loading documents from %s", json_path)
	with json_path.open("r", encoding="utf-8") as handle:
		data = json.load(handle)
	if not isinstance(data, list):
		raise ValueError("Expected a list of documents in the JSON payload")
	logging.info("Loaded %d documents", len(data))
	return data


def guess_doc_type(content: str) -> Optional[str]:
	"""Infer doc type from the trailing token in the scraped payload."""

	trailing = content.strip().splitlines()[-1].strip().lower()
	if not trailing:
		return None
	if "|" in trailing:
		trailing = trailing.split("|")[-1]
	return trailing.replace(" ", "_")


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
	"""Chunk text into overlapping windows to keep embeddings focused."""

	paragraphs = [block.strip() for block in text.split("\n\n") if block.strip()]
	chunks: List[str] = []
	current: List[str] = []
	current_len = 0

	for para in paragraphs:
		para_len = len(para)
		if current_len + para_len > chunk_size and current:
			chunks.append("\n\n".join(current))
			overlap_source = "\n\n".join(current)[-chunk_overlap:]
			current = [overlap_source, para]
			current_len = len(overlap_source) + para_len
		else:
			current.append(para)
			current_len += para_len

	if current:
		chunks.append("\n\n".join(current))

	return chunks


def clean_content(text: str) -> str:
	"""Remove repeated navigation and footer boilerplate from scraped payloads."""

	lines = text.splitlines()
	cleaned: List[str] = []
	for line in lines:
		stripped = line.strip()
		if not stripped:
			if cleaned and cleaned[-1] == "":
				continue
			cleaned.append("")
			continue
		normalized = " ".join(stripped.split())
		if normalized in BOILERPLATE_LINES:
			continue
		if any(normalized.startswith(prefix) for prefix in BOILERPLATE_PREFIXES):
			continue
		if any(normalized.endswith(suffix) for suffix in BOILERPLATE_SUFFIXES):
			continue
		if any(pattern.search(normalized) for pattern in FOOTER_PATTERNS):
			continue
		cleaned.append(stripped)

	while cleaned and cleaned[0] == "":
		cleaned.pop(0)
	while cleaned and cleaned[-1] == "":
		cleaned.pop()

	return "\n".join(cleaned)


@dataclass
class EmbeddingConfig:
	"""Configuration for Azure OpenAI embedding generation."""

	endpoint: str
	deployment: str
	api_key: Optional[str] = None
	api_version: str = "2024-02-15-preview"


class EmbeddingClient:
	"""Wrapper around Azure OpenAI embeddings with retry logic."""

	def __init__(self, config: EmbeddingConfig, *, credential: Optional[DefaultAzureCredential] = None, max_retries: int = 5) -> None:
		client_kwargs: Dict[str, object] = {
			"azure_endpoint": config.endpoint,
			"api_version": config.api_version,
		}
		if config.api_key:
			client_kwargs["api_key"] = config.api_key
		elif credential is not None:
			def token_provider() -> str:
				return credential.get_token("https://cognitiveservices.azure.com/.default").token

			client_kwargs["azure_ad_token_provider"] = token_provider
		else:
			raise RuntimeError("Provide either AZURE_OPENAI_API_KEY or enable Azure AD credential for embeddings.")

		self._client = AzureOpenAI(**client_kwargs)
		self._deployment = config.deployment
		self._max_retries = max_retries
		self._logger = logging.getLogger(self.__class__.__name__)

	def embed(self, text: str) -> List[float]:
		"""Generate a vector embedding with exponential backoff on throttling."""

		delay = 1.0
		for attempt in range(1, self._max_retries + 1):
			try:
				start = time.perf_counter()
				response = self._client.embeddings.create(
					input=text,
					model=self._deployment,
				)
				elapsed = time.perf_counter() - start
				self._logger.debug("Embedding generated in %.2fs", elapsed)
				return response.data[0].embedding
			except Exception as exc:  # pragma: no cover - network error path
				if attempt == self._max_retries:
					raise
				self._logger.warning(
					"Embedding attempt %d failed (%s). Retrying in %.1fs",
					attempt,
					exc,
					delay,
				)
				time.sleep(delay)
				delay *= 2

class AzureIdentityTokenCallback(OIDCCallback):
    """MongoDB OIDC callback that returns an Azure AD access token."""

    def __init__(self, credential: DefaultAzureCredential) -> None:
        self._credential = credential

    def fetch(self, context: OIDCCallbackContext) -> OIDCCallbackResult:  # pragma: no cover - network interaction
        token = self._credential.get_token("https://ossrdbms-aad.database.windows.net/.default").token
        return OIDCCallbackResult(access_token=token)


def normalize_similarity(similarity: str) -> str:
    """Map user-provided similarity string to Cosmos-supported keyword."""

    mapping = {
        "cos": "COS",
        "cosine": "COS",
        "cosine_distance": "COS",
        "l2": "L2",
        "euclidean": "L2",
        "ip": "IP",
        "innerproduct": "IP",
        "inner_product": "IP",
        "dot": "IP",
        "dotproduct": "IP",
    }
    key = similarity.lower().strip()
    if key not in mapping:
        raise ValueError(
            f"Unsupported similarity metric '{similarity}'. Choose from cosine, euclidean/L2, or inner product/IP."
        )
    return mapping[key]


def ensure_collection(
    database: pymongo.database.Database,
    collection_name: str,
    partition_key: str,
) -> Collection:
    """Create collection with hashed partition key if it does not already exist."""

    if collection_name in database.list_collection_names():
        logging.info("Collection %s already exists", collection_name)
        return database[collection_name]

    logging.info("Creating collection %s", collection_name)
    try:
        database.create_collection(collection_name)
    except OperationFailure as exc:
        if "already exists" not in str(exc).lower():
            raise
    return database[collection_name]



def ensure_indexes(
	collection: Collection,
	vector_field: str,
	dimensions: int,
	similarity: str,
	index_kind: str,
	index_options: Dict[str, int],
) -> None:
	"""Ensure vector and supporting indexes exist."""

	logging.info("Ensuring supporting indexes on %s", collection.full_name)
	collection.create_index([("docId", pymongo.HASHED)])
	collection.create_index([("url", pymongo.ASCENDING)])
	collection.create_index([("doc_type", pymongo.ASCENDING)])

	cosmos_similarity = normalize_similarity(similarity)
	index_suffix = index_kind.replace("vector-", "")
	index_name = f"{vector_field}_{index_suffix}"
	cosmos_options = {
		"kind": index_kind,
		"dimensions": dimensions,
		"similarity": cosmos_similarity,
	}
	cosmos_options.update(index_options)

	existing_index = None
	indexes_to_remove: List[str] = []
	for index in collection.list_indexes():
		key_spec = index.get("key", {})
		if key_spec.get(vector_field) != "cosmosSearch":
			continue
		name = index.get("name")
		if name == index_name:
			existing_index = index
		else:
			indexes_to_remove.append(name)

	for stale_name in indexes_to_remove:
		logging.info("Dropping stale vector index %s", stale_name)
		collection.drop_index(stale_name)

	if existing_index:
		options = existing_index.get("cosmosSearchOptions", {})
		if all(options.get(key) == value for key, value in cosmos_options.items()):
			logging.info("Vector index %s already matches requested configuration", index_name)
			return
		logging.info("Dropping vector index %s due to configuration change", index_name)
		collection.drop_index(index_name)

	index_command = {
		"createIndexes": collection.name,
		"indexes": [
			{
				"name": index_name,
				"key": {vector_field: "cosmosSearch"},
				"cosmosSearchOptions": cosmos_options,
			}
		],
	}
	try:
		collection.database.command(index_command)
	except OperationFailure as exc:
		if "already exists" in str(exc):
			logging.info("Vector index %s already exists", index_name)
		else:
			raise


def build_embedding_client_from_env() -> EmbeddingClient:
	"""Create embedding client from environment variables."""

	try:
		config = EmbeddingConfig(
			endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
			deployment=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", os.environ.get("EMBEDDINGS_MODEL_DEPLOYMENT_NAME")),
			api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
			api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
		)
	except KeyError as missing:
		raise RuntimeError(
			"Missing Azure OpenAI configuration. Ensure AZURE_OPENAI_ENDPOINT, "
			"and either AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "
			"EMBEDDINGS_MODEL_DEPLOYMENT_NAME are set."
		) from missing

	if not config.deployment:
		raise RuntimeError(
			"Missing Azure OpenAI embedding deployment. Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT or EMBEDDINGS_MODEL_DEPLOYMENT_NAME."
		)

	credential: Optional[DefaultAzureCredential] = None
	if not config.api_key:
		credential = DefaultAzureCredential()

	return EmbeddingClient(config, credential=credential)

def build_rbac_connection_string(endpoint: str) -> str:
	"""Derive an RBAC-enabled Mongo connection string from a Cosmos endpoint."""

	parsed = urlparse(endpoint.strip())
	host = parsed.netloc or parsed.path or endpoint
	host = host.strip("/")
	if host.startswith("https://"):
		host = host[len("https://") :]
	if host.startswith("http://"):
		host = host[len("http://") :]
	if ":" in host:
		host = host.split(":", 1)[0]
	if not host:
		raise ValueError("Invalid COSMOS_DB_ENDPOINT; unable to identify host")

	if host.endswith(".mongocluster.cosmos.azure.com"):
		return (
			f"mongodb+srv://{host}/?tls=true&authMechanism=MONGODB-OIDC&"
			"retrywrites=false&maxIdleTimeMS=120000"
		)

	return (
		f"mongodb://{host}:10255/?ssl=true&replicaSet=globaldb&retrywrites=false&"
		"maxIdleTimeMS=120000&authMechanism=MONGODB-OIDC"
	)


def connect_to_cosmos(connection_string: str, database_name: str) -> pymongo.database.Database:
	"""Create a MongoClient configured for Microsoft Entra (RBAC) authentication."""

	logging.info("Connecting to Cosmos DB database %s", database_name)
	credential = DefaultAzureCredential()
	auth_properties = {"OIDC_CALLBACK": AzureIdentityTokenCallback(credential)}
	client = pymongo.MongoClient(
		connection_string,
		appname="venmo-ingest",
		retryWrites=False,
		authMechanismProperties=auth_properties,
	)
	return client[database_name]


def hash_doc_id(url: str) -> str:
	"""Generate stable document identifier from url."""

	return hashlib.sha256(url.encode("utf-8")).hexdigest()


def build_payload(
	base_doc: dict,
	chunk: str,
	chunk_index: int,
	embedding: List[float],
) -> dict:
	"""Materialize payload document for Cosmos DB."""

	doc_id = hash_doc_id(base_doc["url"])
	payload = {
		"_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{base_doc['url']}#{chunk_index}")),
		"docId": doc_id,
		"chunkIndex": chunk_index,
		"url": base_doc["url"],
		"doc_type": base_doc.get("doc_type"),
		"source": "venmo_help_center",
		"content": chunk,
		"embedding": embedding,
		"metadata": {
			"title": base_doc.get("title"),
			"raw_type": base_doc.get("raw_type"),
		},
	}
	return payload


def align_embedding_dimensions(vector: List[float], target: Optional[int]) -> List[float]:
	"""Resize embedding to the target dimensionality by truncating or zero-padding."""

	if target is None or len(vector) == target:
		return vector
	if len(vector) > target:
		return vector[:target]
	if len(vector) < target:
		return vector + [0.0] * (target - len(vector))
	return vector


def resolve_index_options(args: argparse.Namespace) -> Dict[str, int]:
	"""Build per-algorithm vector index configuration."""

	if args.index_kind == "vector-ivf":
		num_lists = max(1, args.ivf_num_lists)
		return {"numLists": num_lists}
	if args.index_kind == "vector-hnsw":
		return {
			"m": max(2, args.hnsw_m),
			"efConstruction": max(4, args.hnsw_ef_construction),
		}
	if args.index_kind == "vector-diskann":
		return {
			"maxDegree": max(4, args.diskann_max_degree),
			"lBuild": max(1, args.diskann_lbuild),
		}
	return {}


def ingest_documents(
	documents: Iterable[dict],
	collection: Collection,
	embedder: EmbeddingClient,
	chunk_size: int,
	chunk_overlap: int,
	vector_field: str,
	similarity: str,
	requested_dimensions: Optional[int],
	index_kind: str,
	index_options: Dict[str, int],
) -> None:
	"""Chunk, embed, and upsert documents into Cosmos DB."""

	operations = []
	total_chunks = 0
	index_ready = False
	actual_dimensions: Optional[int] = requested_dimensions

	if requested_dimensions and requested_dimensions > COSMOS_VECTOR_DIMENSION_LIMIT:
		raise ValueError(
			f"Requested embedding dimensionality {requested_dimensions} exceeds the Cosmos DB limit of {COSMOS_VECTOR_DIMENSION_LIMIT}."
		)

	for raw_doc in documents:
		doc_type = guess_doc_type(raw_doc.get("content", ""))
		normalized_doc = {
			"url": raw_doc.get("url"),
			"content": clean_content(raw_doc.get("content", "")),
			"doc_type": doc_type,
			"raw_type": doc_type,
			"title": raw_doc.get("title"),
		}

		if not normalized_doc["content"]:
			logging.debug("Skipping document %s due to empty content after cleanup", normalized_doc["url"])
			continue

		chunks = chunk_text(normalized_doc["content"], chunk_size, chunk_overlap)
		logging.debug("Document %s -> %d chunks", normalized_doc["url"], len(chunks))

		for idx, chunk in enumerate(chunks):
			embedding = embedder.embed(chunk)
			if requested_dimensions is None and len(embedding) > COSMOS_VECTOR_DIMENSION_LIMIT:
				logging.warning(
					"Embedding dimensionality %d exceeds Cosmos DB vector limit %d. Truncating to fit.",
					len(embedding),
					COSMOS_VECTOR_DIMENSION_LIMIT,
				)
				requested_dimensions = COSMOS_VECTOR_DIMENSION_LIMIT
			aligned_embedding = align_embedding_dimensions(embedding, requested_dimensions)
			if not index_ready:
				actual_dimensions = len(aligned_embedding)
				if requested_dimensions and requested_dimensions != actual_dimensions:
					logging.warning(
						"Embedding dimensionality mismatch: requested %d but model produced %d.",
						requested_dimensions,
						actual_dimensions,
					)
				ensure_indexes(collection, vector_field, actual_dimensions, similarity, index_kind, index_options)
				index_ready = True
			payload = build_payload(normalized_doc, chunk, idx, aligned_embedding)
			operations.append(pymongo.UpdateOne(
				{"_id": payload["_id"]},
				{"$set": payload},
				upsert=True,
			))
			total_chunks += 1

		if len(operations) >= 50:
			logging.info("Writing batch of %d documents", len(operations))
			collection.bulk_write(operations, ordered=False)
			operations.clear()

	if operations:
		logging.info("Writing final batch of %d documents", len(operations))
		collection.bulk_write(operations, ordered=False)

	logging.info("Ingested %d chunks", total_chunks)

	if not index_ready and actual_dimensions:
		ensure_indexes(collection, vector_field, actual_dimensions, similarity, index_kind, index_options)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	"""Parse CLI arguments."""

	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input", default="data/venmo_help_docs.json", help="Path to JSON payload")
	parser.add_argument(
		"--database",
		default=os.environ.get("AZURE_COSMOS_DATABASE_NAME", os.environ.get("COSMOS_DATABASE", "venmo")),
	)
	parser.add_argument(
		"--collection",
		default=os.environ.get("AZURE_COSMOS_COLLECTION_NAME", os.environ.get("COSMOS_COLLECTION", "help_docs")),
	)
	parser.add_argument(
		"--partition-key",
		default=os.environ.get("AZURE_COSMOS_PARTITION_KEY", os.environ.get("COSMOS_PARTITION_KEY", "docId")),
	)
	parser.add_argument(
		"--connection-string",
		default=os.environ.get("AZURE_COSMOS_CONNECTION_STRING", os.environ.get("COSMOS_CONNECTION_STRING")),
	)
	parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
	parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
	env_dims = os.environ.get("EMBEDDINGS_DIMENSIONS", os.environ.get("EMBEDDING_DIMENSIONS"))
	parser.add_argument(
		"--embedding-dimensions",
		type=int,
		default=int(env_dims) if env_dims else None,
		help="Expected embedding dimensionality. Defaults to environment variable if set; otherwise inferred from the first embedding.",
	)
	parser.add_argument("--vector-field", default="embedding")
	parser.add_argument("--similarity", default=DEFAULT_SIMILARITY)
	parser.add_argument(
		"--index-kind",
		choices=["vector-ivf", "vector-hnsw", "vector-diskann"],
		default=os.environ.get("VECTOR_INDEX_KIND", DEFAULT_INDEX_KIND),
	)
	parser.add_argument(
		"--ivf-num-lists",
		type=int,
		default=int(os.environ.get("VECTOR_INDEX_IVF_NUM_LISTS", "1")),
		help="Number of IVF lists (clusters). Only used when --index-kind=vector-ivf.",
	)
	parser.add_argument(
		"--hnsw-m",
		type=int,
		default=int(os.environ.get("VECTOR_INDEX_HNSW_M", "16")),
		help="Maximum connections per node for HNSW. Only used when --index-kind=vector-hnsw.",
	)
	parser.add_argument(
		"--hnsw-ef-construction",
		type=int,
		default=int(os.environ.get("VECTOR_INDEX_HNSW_EF_CONSTRUCTION", "64")),
		help="efConstruction parameter for HNSW. Only used when --index-kind=vector-hnsw.",
	)
	parser.add_argument(
		"--diskann-max-degree",
		type=int,
		default=int(os.environ.get("VECTOR_INDEX_DISKANN_MAX_DEGREE", "20")),
		help="maxDegree parameter for DiskANN. Only used when --index-kind=vector-diskann.",
	)
	parser.add_argument(
		"--diskann-lbuild",
		type=int,
		default=int(os.environ.get("VECTOR_INDEX_DISKANN_LBUILD", "10")),
		help="lBuild parameter for DiskANN. Only used when --index-kind=vector-diskann.",
	)
	parser.add_argument("--verbose", action="store_true")
	args = parser.parse_args(argv)

	if not args.connection_string:
		endpoint = os.environ.get("COSMOS_DB_ENDPOINT")
		if endpoint:
			args.connection_string = build_rbac_connection_string(endpoint)

	if not args.connection_string:
		parser.error("Provide a connection string via --connection-string, AZURE_COSMOS_CONNECTION_STRING, or COSMOS_DB_ENDPOINT")

	return args


def main(argv: Optional[List[str]] = None) -> int:
	args = parse_args(argv)
	configure_logging(args.verbose)

	embedder = build_embedding_client_from_env()

	database = connect_to_cosmos(args.connection_string, args.database)
	collection = ensure_collection(database, args.collection, args.partition_key)
	documents = load_documents(Path(args.input))
	index_options = resolve_index_options(args)
	logging.info("Configuring vector index kind=%s with options=%s", args.index_kind, index_options)
	ingest_documents(
		documents,
		collection,
		embedder,
		args.chunk_size,
		args.chunk_overlap,
		args.vector_field,
		args.similarity,
		args.embedding_dimensions,
		args.index_kind,
		index_options,
	)

	logging.info("Ingestion completed successfully")
	return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
	sys.exit(main())

