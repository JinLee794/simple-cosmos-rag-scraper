# Simple Comsos RAG Web Scraper

Utilities for scraping Venmo Help Center content, chunking and embedding it with Azure OpenAI, loading it into Azure Cosmos DB for MongoDB vCore (with vector search), and then probing or chatting against the resulting vector store.

## Prerequisites
- Python 3.10+ plus the packages `requests`, `beautifulsoup4`, `python-dotenv`, `pymongo`, `azure-identity`, `openai`, `opentelemetry-api`, and `opentelemetry-sdk`. Example install:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install requests beautifulsoup4 python-dotenv pymongo azure-identity openai opentelemetry-api opentelemetry-sdk
  ```
- Azure OpenAI resource with an embeddings deployment.
- Azure Cosmos DB for MongoDB vCore cluster with vector search enabled and Microsoft Entra (RBAC) access via `DefaultAzureCredential` (or an explicit Mongo connection string).
- `az login` (or equivalent) on the machine that runs Cosmos-connected scripts so `DefaultAzureCredential` can fetch tokens.

## Environment Setup
1. Copy the template and fill in all required values:
   ```bash
   cp .env.sample .env
   ```
2. Update `.env` so it contains the Azure credentials plus the scraper configuration. The scripts read the variables below (first value = preferred name, second = accepted alias if the template differs):

| Purpose | Variable(s) |
| --- | --- |
| Azure OpenAI endpoint + key | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION` |
| Embedding deployment name | `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` (or `EMBEDDINGS_MODEL_DEPLOYMENT_NAME`) |
| Cosmos DB connection | `AZURE_COSMOS_CONNECTION_STRING` (or `COSMOS_CONNECTION_STRING`) <br>OR set `COSMOS_DB_ENDPOINT` so the scripts derive an RBAC-enabled string |
| Cosmos DB metadata | `AZURE_COSMOS_DATABASE_NAME`/`COSMOS_DATABASE`, `AZURE_COSMOS_COLLECTION_NAME`/`COSMOS_COLLECTION`, optional `AZURE_COSMOS_PARTITION_KEY` |
| Optional embedding shape | `EMBEDDINGS_DIMENSIONS` (or `EMBEDDING_DIMENSIONS`) |
| Scraper controls | `SCRAPE_START_URL` and `SCRAPE_OUTPUT_PATH` *(the `.env.sample` uses `SCRAPER_*`; define both spellings or rename the keys to `SCRAPE_*` so `1.scraper.py` can read them)* |

> Tip: Set `SCRAPE_OUTPUT_PATH` to `data/venmo_help_docs.json` so the ingestion script can read the scraper output without extra flags.

## Workflow (run in order)

### 1. Scrape the Venmo Help Center
`1.scraper.py` performs a breadth-first crawl that stays on the seed domain, removes boilerplate text, and caches the results as JSON.

```bash
python 1.scraper.py
```

- Reads `SCRAPE_START_URL` for the seed page and writes cleaned documents to `SCRAPE_OUTPUT_PATH`.
- Reuses the cached JSON on subsequent runs; delete the file to force a fresh crawl.

### 2. Ingest content into Cosmos DB
`2.cosmos_data_ingestion.py` chunks each scraped document, generates embeddings with Azure OpenAI (retrying on throttling), and upserts the payload into Cosmos DB while ensuring vector indexes exist.

```bash
python 2.cosmos_data_ingestion.py \
  --input data/venmo_help_docs.json \
  --database "$AZURE_COSMOS_DATABASE_NAME" \
  --collection "$AZURE_COSMOS_COLLECTION_NAME" \
  --connection-string "$AZURE_COSMOS_CONNECTION_STRING" \
  --chunk-size 1200 \
  --chunk-overlap 200 \
  --vector-field embedding \
  --similarity cosine \
  --index-kind vector-hnsw \
  --verbose
```

- Omit `--connection-string` if you prefer to set `COSMOS_DB_ENDPOINT` and rely on RBAC.
- Use `--embedding-dimensions` if Cosmos expects a specific vector length (otherwise the first embedding determines it).
- The script batches writes (50 docs per bulk upsert) and only recreates vector indexes when necessary.

### 3. Probe retrieval latency
`3.retrieval_latency_probe.py` embeds each query, performs repeated vector searches, and summarizes the latency distribution. `--query` can be supplied multiple times, or use `--query-file` for a newline-delimited list. Enable verbose mode to dump the top hits for each query.

```bash
python 3.retrieval_latency_probe.py \
  --query "How do I transfer money from Venmo to my bank?" \
  --query "What are Venmo transfer limits?" \
  --iterations 5 \
  --warmup 1 \
  --enable-tracing \
  --verbose > probe.json
```

- Output prints summary stats to stdout; when piping to a file (as above) you get both the logs and the structured verbose output.
- `--enable-tracing` emits OpenTelemetry spans (useful when correlating Cosmos/OpenAI latency).

### 4. Chat against the knowledge base
`4.chat_client.py` is a thin CLI wrapper over `CosmosVectorRetriever`. Provide a query via `--query` or skip the flag to be prompted interactively.

```bash
python 4.chat_client.py \
  --query "How do I reset my Venmo PIN?" \
  --top-k 5 \
  --num-candidates 40 \
  --vector-field embedding \
  --similarity cosine \
  --verbose
```

- Results include the score, inferred doc type, URL, and a snippet of the chunk content.
- All Cosmos/Azure credentials are resolved the same way as in the latency probe, so keep `.env` loaded or export the variables in your shell session.

## Operational Notes
- `DefaultAzureCredential` tries multiple identity sources (managed identity, Azure CLI login, VS Code, etc.). Run `az login` first when using local CLI credentials.
- If the ingestion job reports a dimension mismatch, capture the `embedding.dimension` logged by the script and pass it via `--embedding-dimensions` so Cosmos can pad/truncate consistently.
- The retriever automatically falls back from `$vectorSearch` to `$search` if the cluster does not yet support the newer stage.
- Keep scraped data under version control ignore rules (see `.gitignore`) to avoid committing large JSON payloads or secrets.
