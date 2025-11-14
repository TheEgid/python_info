from __future__ import annotations

import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Set, Tuple

import clickhouse_connect
import numpy as np
from llama_index.core import Settings
from llama_index.core.schema import Document, TextNode

from others.frida import FridaEmbedding
from others.wiki_scraper import run_scraper_separate_files  # noqa: F401

_morph_analyzer = None


def _get_morph_analyzer():  # noqa: ANN202
    """Lazy initialization of morphological analyzer."""
    global _morph_analyzer
    if _morph_analyzer is None:
        import pymorphy3

        _morph_analyzer = pymorphy3.MorphAnalyzer()
    return _morph_analyzer


# Compile regex once at module level
VALID_TABLE_RE = re.compile(r"^[A-Za-z0-9_]+$")

# Constants
DEFAULT_BATCH_SIZE = 64
DEFAULT_EMBEDDING_WORKERS = 4
MAX_TEXT_LENGTH_FOR_HASH = 2000

# ----------------------------- Utilities ---------------------------------


def lemmatize_text(text: str) -> str:
    """Return simple lemma sequence. Keep separate from embedding pipeline
    (why: lemmatization can harm embedding semantic fidelity)."""
    morph = _get_morph_analyzer()
    words = (w for w in text.split() if w.strip())
    lemmas = [morph.parse(w)[0].normal_form for w in words]
    return " ".join(lemmas)


@lru_cache(maxsize=10000)
def _hash_string(s: str) -> str:
    """Cached hash computation for frequently used strings."""
    return hashlib.md5(s.encode()).hexdigest()


def generate_stable_doc_id(doc: Document) -> str:
    """Create a stable id using url/file_path or content hash.

    Priority order:
    1. URL (most stable)
    2. File path
    3. Content hash (fallback)
    """
    url = doc.metadata.get("url")
    if url:
        return f"doc_url_{_hash_string(str(url))}"

    file_path = doc.metadata.get("file_path")
    if file_path:
        return f"doc_file_{_hash_string(str(file_path))}"

    content = (doc.text or "")[:MAX_TEXT_LENGTH_FOR_HASH]
    return f"doc_content_{_hash_string(content)}"


def validate_table_name(name: str) -> str:
    """Ensure table name is safe (prevents SQL injection)."""
    if not VALID_TABLE_RE.match(name):
        raise ValueError(f"Invalid table name '{name}'. Only letters, digits and underscores allowed")
    return name


def is_finite_vector(vec: Iterable[float]) -> bool:
    """Check all values are finite numbers using numpy for speed."""
    try:
        arr = np.asarray(vec, dtype=np.float32)
        return bool(np.all(np.isfinite(arr)))
    except (ValueError, TypeError):
        return False


def normalize_vector(vec: List[float]) -> List[float]:
    """Normalize vector to unit length for cosine distance optimization."""
    arr = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        return (arr / norm).tolist()
    return vec


# ------------------------- ClickHouse Vector Store ------------------------


class ClickHouseVectorStore:
    """Light wrapper for ClickHouse vector retrieval and metadata operations."""

    def __init__(
        self, client: clickhouse_connect.driver.Client, table_name: str, normalize_vectors: bool = True
    ) -> None:
        self.client = client
        self.table_name = validate_table_name(table_name)
        self.normalize_vectors = normalize_vectors
        # llama_index expects attribute
        self.stores_text = True
        self._count_cache: Optional[Tuple[int, float]] = None  # (count, timestamp)

    def count(self, use_cache: bool = True, cache_ttl: float = 60.0) -> int:
        """Get document count with optional caching."""
        import time

        if use_cache and self._count_cache:
            cached_count, cached_time = self._count_cache
            if time.time() - cached_time < cache_ttl:
                return cached_count

        try:
            q = f"SELECT count() FROM {self.table_name}"
            count = int(self.client.command(q))
            self._count_cache = (count, time.time())
            return count
        except Exception as e:
            logging.error("Failed to get count from %s: %s", self.table_name, e)
            return 0

    def search_by_cosine(
        self, query_embedding: List[float], limit: int = 10, filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search by cosine distance with optional metadata filtering."""
        if self.normalize_vectors:
            query_embedding = normalize_vector(query_embedding)

        # Build WHERE clause for metadata filtering
        where_clause = ""
        if filter_metadata:
            conditions = []
            for key, value in filter_metadata.items():
                # Safe JSON extraction using JSONExtractString
                conditions.append(f"JSONExtractString(metadata, '{key}') = '{value}'")
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)

        q = f"""
        SELECT
            id,
            text,
            metadata,
            cosineDistance(embedding, %(q_emb)s) AS dist
        FROM {self.table_name}
        {where_clause}
        ORDER BY dist
        LIMIT %(limit)s
        """

        try:
            res = self.client.query(q, {"q_emb": query_embedding, "limit": limit})
            # Compatible with Python < 3.10 (no strict parameter)
            return [dict(zip(res.column_names, row)) for row in res.result_rows]  # noqa: B905
        except Exception as e:
            logging.error("Vector search failed in %s: %s", self.table_name, e)
            return []

    def delete_by_ids(self, doc_ids: List[str]) -> int:
        """Delete documents by IDs. Returns number of deleted rows."""
        if not doc_ids:
            return 0

        try:
            # Use ALTER TABLE DELETE for efficient deletion
            ids_str = ", ".join(f"'{doc_id}'" for doc_id in doc_ids)
            q = f"ALTER TABLE {self.table_name} DELETE WHERE id IN ({ids_str})"
            self.client.command(q)

            # Clear cache
            self._count_cache = None

            logging.info("Deleted %d documents from %s", len(doc_ids), self.table_name)
            return len(doc_ids)
        except Exception as e:
            logging.error("Failed to delete documents: %s", e)
            return 0


# ------------------------- ClickHouse helpers -----------------------------


@contextmanager
def get_clickhouse_client(
    host: str = "192.168.1.77",
    port: int = 8123,
    username: str = "default",
    password: str = "",
    database: str = "default",
) -> Generator[clickhouse_connect.driver.Client, None, None]:
    """Create and sanity-check clickhouse-connect client with context manager.

    Usage:
        with get_clickhouse_client() as client:
            client.query(...)
    """
    client = None
    try:
        client = clickhouse_connect.get_client(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
            connect_timeout=10,
            send_receive_timeout=30,
        )
        # Verify connection
        client.command("SELECT 1")
        logging.info("Connected to ClickHouse %s:%s (database: %s)", host, port, database)
        yield client
    except Exception as e:
        logging.error("Failed to connect to ClickHouse: %s", e)
        raise
    finally:
        if client:
            client.close()


def create_table(
    client: clickhouse_connect.driver.Client,
    table_name: str = "novaya",
    vector_dimension: Optional[int] = None,
    create_index: bool = True,
) -> None:
    """Create table with optimized schema and optional vector index."""
    table_name = validate_table_name(table_name)

    # Build dimension constraint if specified
    dimension_constraint = ""
    if vector_dimension:
        dimension_constraint = f"CONSTRAINT check_embedding_dim CHECK length(embedding) = {vector_dimension}"

    sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id String,
        text String,
        embedding Array(Float32) {dimension_constraint},
        metadata String,
        created_at DateTime DEFAULT now()
    ) ENGINE = MergeTree()
    ORDER BY (id, created_at)
    SETTINGS index_granularity = 8192
    """

    client.command(sql)
    logging.info("Table '%s' created/verified", table_name)

    # Create additional indexes for better query performance
    if create_index:
        try:
            # Index on metadata fields (if using JSON queries frequently)
            client.command(
                f"ALTER TABLE {table_name} "
                f"ADD INDEX IF NOT EXISTS idx_metadata metadata TYPE ngrambf_v1(4, 512, 2, 0) GRANULARITY 1"
            )
        except Exception as e:
            logging.debug("Index creation skipped or failed: %s", e)


def get_existing_doc_ids(client: clickhouse_connect.driver.Client, table_name: str = "novaya") -> Set[str]:
    """Get existing document IDs efficiently using set."""
    table_name = validate_table_name(table_name)

    # Check if table exists
    try:
        client.command(f"SELECT 1 FROM {table_name} LIMIT 0")
    except Exception:
        logging.info("Table '%s' does not exist", table_name)
        return set()

    try:
        # Use DISTINCT for deduplication at database level
        res = client.query(f"SELECT DISTINCT id FROM {table_name}")
        return {str(r[0]) for r in res.result_rows if r and r[0]}
    except Exception as e:
        logging.error("Failed to read existing ids: %s", e)
        return set()


def assemble_documents_from_chunks(documents: List[Document]) -> List[Document]:
    """Combine chunked documents by numeric chunk_index. Preserve metadata from first chunk.

    Optimizations:
    - Single pass grouping
    - Efficient string concatenation
    - Preserve all metadata
    """
    chunks_by_url: Dict[str, List[Document]] = {}
    standalone: List[Document] = []

    # Single pass grouping
    for doc in documents:
        url = doc.metadata.get("url")
        if url:
            chunks_by_url.setdefault(url, []).append(doc)
        else:
            standalone.append(doc)

    assembled: List[Document] = []

    for url, chunks in chunks_by_url.items():
        if not chunks:
            continue

        # Sort by chunk_index
        def _key(d: Document) -> int:
            try:
                return int(d.metadata.get("chunk_index", 0))
            except (ValueError, TypeError):
                return 0

        sorted_chunks = sorted(chunks, key=_key)

        # Efficient string concatenation using join
        combined = "\n\n".join(c.text for c in sorted_chunks if c.text)

        # Preserve all metadata from first chunk
        meta = sorted_chunks[0].metadata.copy()
        meta.update(
            {
                "chunks_count": len(sorted_chunks),
                "assembled_from_chunks": True,
                "url": url,
                "chunk_indices": [c.metadata.get("chunk_index") for c in sorted_chunks],
            }
        )

        assembled.append(Document(text=combined, metadata=meta))

    return assembled + standalone


# ----------------------- Insertion / Loading logic -----------------------


def _batch(iterable: Iterable[Any], size: int) -> Generator[List[Any], None, None]:
    """Efficiently batch an iterable into chunks of given size."""
    batch: List[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _get_text_embeddings_for_texts(
    embed_model: FridaEmbedding, texts: List[str], parallel: bool = False, max_workers: int = DEFAULT_EMBEDDING_WORKERS
) -> List[List[float]]:
    """Get embeddings with batch API and parallel processing fallback.

    Priority:
    1. Batch API (get_text_embeddings)
    2. Parallel single calls
    3. Sequential fallback
    """
    if not texts:
        return []

    # Try batch API first
    if hasattr(embed_model, "get_text_embeddings"):
        try:
            return embed_model.get_text_embeddings(texts)
        except Exception as e:
            logging.warning("Batch embedding failed, using fallback: %s", e)

    if hasattr(embed_model, "get_batch_embeddings"):
        try:
            return embed_model.get_batch_embeddings(texts)
        except Exception as e:
            logging.warning("Batch embedding failed, using fallback: %s", e)

    # Parallel fallback for speed
    if parallel and len(texts) > 1:
        embeddings: List[Optional[List[float]]] = [None] * len(texts)

        def _embed_single(idx: int, text: str) -> Tuple[int, List[float]]:
            if hasattr(embed_model, "get_text_embedding"):
                return idx, embed_model.get_text_embedding(text)
            return idx, embed_model._get_text_embedding(text)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_embed_single, idx, text): idx for idx, text in enumerate(texts)}

            for future in as_completed(futures):
                try:
                    idx, emb = future.result()
                    embeddings[idx] = emb
                except Exception as e:
                    idx = futures[future]
                    logging.error("Failed to embed text at index %d: %s", idx, e)
                    embeddings[idx] = []

        return [emb if emb is not None else [] for emb in embeddings]

    # Sequential fallback
    embeddings_seq: List[List[float]] = []
    for text in texts:
        try:
            if hasattr(embed_model, "get_text_embedding"):
                embeddings_seq.append(embed_model.get_text_embedding(text))
            else:
                embeddings_seq.append(embed_model._get_text_embedding(text))
        except Exception as e:
            logging.error("Failed to embed text: %s", e)
            embeddings_seq.append([])

    return embeddings_seq


def insert_documents_to_clickhouse(
    documents: List[Document],
    client: clickhouse_connect.driver.Client,
    table_name: str = "novaya",
    batch_size: int = DEFAULT_BATCH_SIZE,
    lemmatize_for_index: bool = False,
    normalize_vectors: bool = True,
    parallel_embedding: bool = True,
) -> Tuple[Optional[ClickHouseVectorStore], List[TextNode]]:
    """Insert documents in batches with embeddings. Returns store and created nodes.

    Optimizations:
    - Parallel embedding processing
    - Bulk JSON serialization
    - Efficient batching
    - Vector normalization for better search performance

    Args:
        documents: List of documents to insert
        client: ClickHouse client
        table_name: Target table name
        batch_size: Number of documents per batch
        lemmatize_for_index: Whether to lemmatize text before embedding
        normalize_vectors: Normalize embeddings to unit vectors
        parallel_embedding: Use parallel processing for embeddings
    """
    table_name = validate_table_name(table_name)

    if not documents:
        logging.warning("No documents to insert")
        return None, []

    create_table(client, table_name)

    # Prepare embedding model and register with llama_index Settings
    embed_model = FridaEmbedding()
    Settings.embed_model = embed_model

    nodes: List[TextNode] = []
    total_inserted = 0

    # Filter and prepare documents
    prepared: List[Tuple[Document, str, str]] = []
    for doc in documents:
        text = (doc.text or "").strip()
        if not text:
            continue

        doc_id = getattr(doc, "doc_id", None) or generate_stable_doc_id(doc)
        meta = doc.metadata.copy()
        meta["doc_id"] = doc_id

        # Preprocess text
        processed = lemmatize_text(text) if lemmatize_for_index else text
        prepared.append((doc, doc_id, processed))

    if not prepared:
        logging.error("No valid documents after preprocessing")
        return None, []

    # Process in batches
    for batch_idx, batch_items in enumerate(_batch(prepared, batch_size)):
        texts = [item[2] for item in batch_items]

        try:
            embeddings = _get_text_embeddings_for_texts(embed_model, texts, parallel=parallel_embedding)
        except Exception as e:
            logging.error("Embedding batch %d failed: %s", batch_idx, e)
            continue

        # Prepare rows for bulk insert
        rows: List[Tuple[str, str, List[float], str]] = []
        batch_nodes: List[TextNode] = []

        for (doc, doc_id, processed), emb in zip(batch_items, embeddings):  # noqa: B905
            if not emb or not is_finite_vector(emb):
                logging.warning("Skipping doc %s due to invalid embedding", doc_id)
                continue

            # Normalize if requested
            if normalize_vectors:
                emb = normalize_vector(emb)

            metadata = doc.metadata.copy()
            metadata.setdefault("source", "unknown")
            metadata["doc_id"] = doc_id

            # Serialize metadata once
            metadata_json = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))

            rows.append((doc_id, processed, [float(x) for x in emb], metadata_json))

            batch_nodes.append(
                TextNode(text=processed, embedding=[float(x) for x in emb], metadata=metadata, id_=doc_id)
            )

        if not rows:
            logging.warning("Batch %d produced no valid rows", batch_idx)
            continue

        # Bulk insert
        try:
            client.insert(table_name, rows, column_names=["id", "text", "embedding", "metadata"])
            total_inserted += len(rows)
            nodes.extend(batch_nodes)
            logging.info(
                "Inserted batch %d: %d rows into %s (total: %d)", batch_idx, len(rows), table_name, total_inserted
            )
        except Exception as e:
            logging.exception("Failed to insert batch %d: %s", batch_idx, e)

    if total_inserted == 0:
        logging.error("No documents were successfully inserted")
        return None, []

    store = ClickHouseVectorStore(client, table_name, normalize_vectors=normalize_vectors)
    logging.info("Insertion complete: %d documents inserted, %d nodes created", total_inserted, len(nodes))
    return store, nodes


def load_existing_data(
    client: clickhouse_connect.driver.Client, table_name: str = "novaya", limit: Optional[int] = None
) -> Tuple[Optional[ClickHouseVectorStore], Optional[List[TextNode]]]:
    """Load existing data from ClickHouse with optional limit."""
    table_name = validate_table_name(table_name)

    # Check table exists
    try:
        client.command(f"SELECT 1 FROM {table_name} LIMIT 0")
    except Exception:
        logging.info("Table '%s' not found", table_name)
        return None, None

    try:
        # Build query with optional limit
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"""
        SELECT id, text, embedding, metadata
        FROM {table_name}
        ORDER BY created_at DESC
        {limit_clause}
        """

        res = client.query(query)
        nodes: List[TextNode] = []

        for row in res.result_rows:
            doc_id, text, embedding, metadata_str = row

            # Parse metadata
            metadata = {}
            if isinstance(metadata_str, str):
                try:
                    metadata = json.loads(metadata_str)
                except json.JSONDecodeError as e:
                    logging.warning("Failed to parse metadata for %s: %s", doc_id, e)

            node_id = metadata.get("doc_id") or doc_id
            nodes.append(TextNode(text=text, metadata=metadata or {}, embedding=embedding or [], id_=node_id))

        store = ClickHouseVectorStore(client, table_name)
        logging.info("Loaded %d nodes from %s", len(nodes), table_name)
        return store, nodes

    except Exception as e:
        logging.exception("Failed to load data from %s: %s", table_name, e)
        return None, None


def add_documents_to_clickhouse(
    documents: List[Document],
    client: clickhouse_connect.driver.Client,
    table_name: str = "novaya",
    force_recreate: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lemmatize_for_index: bool = False,
    deduplicate: bool = True,
) -> Tuple[Optional[ClickHouseVectorStore], Optional[List[TextNode]]]:
    """Add documents to ClickHouse with deduplication.

    Args:
        documents: Documents to add
        client: ClickHouse client
        table_name: Target table
        force_recreate: Drop and recreate table
        batch_size: Batch size for insertion
        lemmatize_for_index: Apply lemmatization
        deduplicate: Skip existing documents
    """
    table_name = validate_table_name(table_name)

    if force_recreate:
        try:
            client.command(f"DROP TABLE IF EXISTS {table_name}")
            logging.info("Dropped table %s", table_name)
        except Exception as e:
            logging.debug("Drop failed (expected if table doesn't exist): %s", e)

    create_table(client, table_name)

    # Assemble chunks
    processed = assemble_documents_from_chunks(documents)

    # Deduplicate if requested
    if deduplicate:
        existing = get_existing_doc_ids(client, table_name)
        new_docs = []

        for doc in processed:
            doc_id = getattr(doc, "doc_id", None) or generate_stable_doc_id(doc)
            if doc_id not in existing:
                doc.doc_id = doc_id
                doc.metadata["doc_id"] = doc_id
                new_docs.append(doc)

        logging.info("Deduplication: %d existing, %d new out of %d total", len(existing), len(new_docs), len(processed))
    else:
        new_docs = processed
        for doc in new_docs:
            doc_id = getattr(doc, "doc_id", None) or generate_stable_doc_id(doc)
            doc.doc_id = doc_id
            doc.metadata["doc_id"] = doc_id

    if not new_docs:
        logging.info("No new documents to add")
        return load_existing_data(client, table_name)

    return insert_documents_to_clickhouse(
        new_docs, client, table_name, batch_size=batch_size, lemmatize_for_index=lemmatize_for_index
    )


def initialize_clickhouse_database(
    db_host: str = "192.168.1.77",
    db_port: int = 8123,
    db_user: str = "default",
    db_password: str = "",
    db_name: str = "default",
    table_name: str = "novaya",
    documents: Optional[List[Document]] = None,
    force_recreate: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple[Optional[ClickHouseVectorStore], Optional[List[TextNode]]]:
    """Initialize ClickHouse database and optionally insert documents.

    Returns:
        Tuple of (vector_store, nodes) or (None, None) on failure
    """
    try:
        client = clickhouse_connect.get_client(
            host=db_host, port=db_port, username=db_user, password=db_password, database=db_name
        )

        try:
            if documents:
                return add_documents_to_clickhouse(
                    documents,
                    client,
                    table_name,
                    force_recreate=force_recreate,
                    batch_size=batch_size,
                    deduplicate=True,
                )
            return load_existing_data(client, table_name)
        finally:
            # Always close client
            client.close()

    except Exception as e:
        logging.exception("Initialization failed: %s", e)
        return None, None


# ------------------------- Wiki Article Parser -------------------------


def _extract_wiki_text_from_html(html: str) -> str:
    """Lightweight parser for Wikipedia-like HTML exports.

    Extracts text from common tags and removes boilerplate.
    Uses BeautifulSoup if available, falls back to regex otherwise.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logging.warning("BeautifulSoup not installed — using basic regex extraction")
        # Simple regex fallback
        import re

        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content blocks
    for tag in soup.select("script, style, header, footer, nav, aside, table.infobox, .navbox"):
        tag.decompose()

    # Extract paragraphs and headers
    pieces = []
    for tag in soup.select("h1, h2, h3, h4, p, li"):
        text = tag.get_text(" ", strip=True)
        if text and len(text) > 10:  # Filter very short fragments
            pieces.append(text)

    return "\n\n".join(pieces)


def _load_wiki_articles(folder: str) -> List[Document]:
    """Load Wikipedia-like exported HTML files and parse them into plain text Documents."""
    from pathlib import Path

    docs = []
    base = Path(folder)

    if not base.exists():
        logging.error("Wiki folder not found: %s", folder)
        return []

    html_files = list(base.glob("**/*.html")) + list(base.glob("**/*.htm"))

    for p in html_files:
        try:
            raw_html = p.read_text(encoding="utf-8", errors="ignore")
            parsed = _extract_wiki_text_from_html(raw_html)

            if not parsed.strip() or len(parsed) < 50:
                logging.debug("Skipping empty/short wiki file: %s", p)
                continue

            meta = {
                "file_path": str(p),
                "file_name": p.name,
                "source": "wiki_html",
                "url": None,
            }

            docs.append(Document(text=parsed, metadata=meta))

        except Exception as e:
            logging.error("Failed parsing wiki file %s: %s", p, e)

    logging.info("Loaded %d wiki articles from %s", len(docs), folder)
    return docs


# ------------------------- Document Loading -------------------------


def _load_documents_from_folder(folder: str) -> List[Document]:
    """Load documents from folder supporting multiple formats."""
    docs = []
    base = Path(folder)

    logging.info(base)

    if not base.exists():
        logging.error("Documents folder not found: %s", folder)
        return []

    # Supported extensions
    extensions = {".txt", ".md", ".json", ".html", ".htm"}

    for p in base.glob("**/*"):
        logging.error(p)
        if not p.is_file() or p.suffix.lower() not in extensions:
            continue

        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            logging.info(text)

            if not text.strip() or len(text) < 10:
                logging.debug("Skipping empty/short file: %s", p)
                continue

            meta = {"file_path": str(p), "file_name": p.name, "file_type": p.suffix.lower(), "source": "folder"}

            docs.append(Document(text=text, metadata=meta))

        except Exception as e:
            logging.error("Failed reading file %s: %s", p, e)

    logging.info("Loaded %d documents from %s", len(docs), folder)
    return docs


def initialize_clickhouse(
    documents_source: str = "articles",
    source_type: str = "folder",  # "folder" or "wiki"
    **kwargs,  # noqa: ANN003
) -> Tuple[Optional[ClickHouseVectorStore], Optional[List[TextNode]]]:
    """Backward-compatible wrapper expected by main RAG script.

    Args:
        documents_source: Path to folder with documents
        source_type: Type of documents ("folder" for plain text, "wiki" for HTML)
        **kwargs: Additional arguments passed to initialize_clickhouse_database
    """
    if source_type == "wiki":
        documents = _load_wiki_articles(documents_source)
    else:
        documents = _load_documents_from_folder(documents_source)

    return initialize_clickhouse_database(documents=documents, **kwargs)


# В конец файла main.py добавить:

# ------------------------- CLI и точка запуска -------------------------


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with colors and formatting."""
    import sys

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Color codes for terminal
    colors = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    class ColoredFormatter(logging.Formatter):
        def format(self, record):  # noqa: ANN001, ANN202
            levelname = record.levelname
            if levelname in colors:
                record.levelname = f"{colors[levelname]}{levelname}{colors['RESET']}"
            return super().format(record)

    formatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = [handler]


def print_stats(store: ClickHouseVectorStore, nodes: List[TextNode]) -> None:
    """Print statistics about loaded data."""
    print("\n" + "=" * 60)
    print("📊 DATABASE STATISTICS")
    print("=" * 60)
    print(f"Total documents in DB: {store.count()}")
    print(f"Loaded nodes: {len(nodes)}")

    if nodes:
        # Analyze metadata
        sources = {}
        for node in nodes:
            source = node.metadata.get("source", "unknown")
            sources[source] = sources.get(source, 0) + 1

        print("\n📁 Documents by source:")
        for source, count in sorted(sources.items()):
            print(f"  • {source}: {count}")

        # Embedding stats
        if nodes[0].embedding:
            emb_dim = len(nodes[0].embedding)
            print(f"\n🔢 Embedding dimension: {emb_dim}")

            # Sample quality check
            valid_embeddings = sum(1 for n in nodes[:100] if is_finite_vector(n.embedding or []))
            print(f"✓ Valid embeddings (sample): {valid_embeddings}/100")

    print("=" * 60 + "\n")


def interactive_search(store: ClickHouseVectorStore, embed_model: FridaEmbedding) -> None:
    """Interactive search mode."""
    print("\n" + "=" * 60)
    print("🔍 INTERACTIVE SEARCH MODE")
    print("=" * 60)
    print("Type your queries below. Commands:")
    print("  • 'quit' or 'exit' - exit search mode")
    print("  • 'limit N' - set result limit (default: 5)")
    print("=" * 60 + "\n")

    limit = 5

    while True:
        try:
            query = input("\n🔎 Query: ").strip()

            if not query:
                continue

            if query.lower() in ("quit", "exit", "q"):
                print("👋 Goodbye!")
                break

            if query.lower().startswith("limit "):
                try:
                    limit = int(query.split()[1])
                    print(f"✓ Result limit set to {limit}")
                    continue
                except (ValueError, IndexError):
                    print("❌ Invalid limit format. Use: limit N")
                    continue

            # Generate embedding
            query_emb = embed_model.get_text_embedding(query)

            # Search
            results = store.search_by_cosine(query_emb, limit=limit)

            if not results:
                print("❌ No results found")
                continue

            print(f"\n✨ Found {len(results)} results:\n")

            for i, result in enumerate(results, 1):
                dist = result.get("dist", 0)
                text = result.get("text", "")[:200]
                metadata = result.get("metadata", "{}")

                # Parse metadata
                try:
                    meta_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
                    source = meta_dict.get("source", "unknown")
                    file_name = meta_dict.get("file_name", meta_dict.get("url", "N/A"))
                except:  # noqa: E722
                    source = "unknown"
                    file_name = "N/A"

                print(f"{i}. [Distance: {dist:.4f}] ({source})")
                print(f"   File: {file_name}")
                print(f"   Text: {text}...")
                print()

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logging.error("Search error: %s", e)


def benchmark_search(store: ClickHouseVectorStore, embed_model: FridaEmbedding, num_queries: int = 10) -> None:
    """Run search benchmarks."""
    import time

    print("\n" + "=" * 60)
    print(f"⚡ RUNNING SEARCH BENCHMARK ({num_queries} queries)")
    print("=" * 60)

    test_queries = [
        "What is machine learning?",
        "How does neural network work?",
        "Explain vector embeddings",
        "Database optimization techniques",
        "Python programming best practices",
        "Data science fundamentals",
        "Cloud computing architecture",
        "Artificial intelligence applications",
        "Web development frameworks",
        "Cybersecurity basics",
    ]

    queries = test_queries[:num_queries]

    # Warmup
    warmup_emb = embed_model.get_text_embedding(queries[0])
    store.search_by_cosine(warmup_emb, limit=5)

    # Benchmark
    total_time = 0
    embedding_time = 0
    search_time = 0

    for query in queries:
        # Embedding time
        t0 = time.time()
        query_emb = embed_model.get_text_embedding(query)
        t1 = time.time()
        embedding_time += t1 - t0

        # Search time
        t2 = time.time()
        results = store.search_by_cosine(query_emb, limit=10)  # noqa: F841
        t3 = time.time()
        search_time += t3 - t2

        total_time += t3 - t0

    avg_total = total_time / num_queries
    avg_embedding = embedding_time / num_queries
    avg_search = search_time / num_queries

    print("\n📈 Results:")
    print(f"  • Average total time: {avg_total * 1000:.2f} ms")
    print(f"  • Average embedding time: {avg_embedding * 1000:.2f} ms")
    print(f"  • Average search time: {avg_search * 1000:.2f} ms")
    print(f"  • Queries per second: {1 / avg_total:.2f}")
    print("=" * 60 + "\n")


def export_to_json(store: ClickHouseVectorStore, nodes: List[TextNode], output_file: str = "export.json") -> None:
    """Export data to JSON file."""
    print(f"\n📦 Exporting to {output_file}...")

    data = []
    for node in nodes:
        data.append(
            {
                "id": node.id_,
                "text": node.text,
                "metadata": node.metadata,
                "embedding_dim": len(node.embedding) if node.embedding else 0,
            }
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✓ Exported {len(data)} documents to {output_file}")


def main() -> None:
    """Main CLI entry point."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="ClickHouse Vector Store - Document indexing and search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First-time setup: index documents from folder
  python main.py --action index --source ./documents

  # Index Wikipedia HTML exports
  python main.py --action index --source ./wiki --source-type wiki

  # Check database status
  python main.py --action stats

  # Interactive search
  python main.py --action search

  # Run benchmarks
  python main.py --action benchmark

  # Export to JSON
  python main.py --action export --output data.json

  # Clear database
  python main.py --action clear --table novaya --confirm
        """,
    )

    # Action
    parser.add_argument(
        "--action",
        choices=["index", "search", "benchmark", "export", "stats", "clear", "init"],
        default="stats",
        help="Action to perform (default: stats)",
    )

    # Database connection
    parser.add_argument("--host", default="192.168.1.77", help="ClickHouse host")
    parser.add_argument("--port", type=int, default=8123, help="ClickHouse port")
    parser.add_argument("--user", default="default", help="ClickHouse username")
    parser.add_argument("--password", default="", help="ClickHouse password")
    parser.add_argument("--database", default="default", help="ClickHouse database")
    parser.add_argument("--table", default="novaya", help="Table name")

    # Indexing options
    parser.add_argument("--source", help="Source folder for documents")
    parser.add_argument("--source-type", choices=["folder", "wiki"], default="folder", help="Type of source documents")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for indexing")
    parser.add_argument("--force-recreate", action="store_true", help="Drop and recreate table")
    parser.add_argument("--lemmatize", action="store_true", help="Apply lemmatization before embedding")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel embedding processing")

    # Search options
    parser.add_argument("--query", help="Search query (non-interactive mode)")
    parser.add_argument("--limit", type=int, default=5, help="Number of search results")

    # Export options
    parser.add_argument("--output", default="export.json", help="Output file for export")

    # Benchmark options
    parser.add_argument("--num-queries", type=int, default=10, help="Number of benchmark queries")

    # Clear options
    parser.add_argument("--confirm", action="store_true", help="Skip confirmation prompt for destructive operations")

    # Logging
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    try:
        # Connect to ClickHouse
        logging.info("Connecting to ClickHouse %s:%s...", args.host, args.port)
        client = clickhouse_connect.get_client(
            host=args.host, port=args.port, username=args.user, password=args.password, database=args.database
        )

        # Test connection
        try:
            client.command("SELECT 1")
            logging.info("✓ Connection successful")
        except Exception as e:
            logging.error("❌ Connection failed: %s", e)
            print("\n" + "=" * 60)
            print("CONNECTION TROUBLESHOOTING")
            print("=" * 60)
            print(f"Host: {args.host}:{args.port}")
            print(f"User: {args.user}")
            print(f"Database: {args.database}")
            print("\nPossible issues:")
            print("  1. ClickHouse server is not running")
            print("  2. Wrong host/port")
            print("  3. Firewall blocking connection")
            print("  4. Authentication failed")
            print("\nTry:")
            print(f"  clickhouse-client --host {args.host} --port 8123")
            print("=" * 60)
            sys.exit(1)

        # Helper: check if table exists
        def table_exists(table_name: str) -> bool:
            try:
                client.command(f"SELECT 1 FROM {validate_table_name(table_name)} LIMIT 0")
                return True
            except Exception:
                return False

        # Execute action
        if args.action == "init":
            # Initialize empty table
            print(f"\n📋 Initializing table '{args.table}'...")
            create_table(client, args.table)
            print(f"✓ Table '{args.table}' created and ready for indexing")
            print("\nNext steps:")
            print("python main.py --action index --source <folder>")

        elif args.action == "clear":
            # Clear table
            if not table_exists(args.table):
                print(f"ℹ️  Table '{args.table}' does not exist (nothing to clear)")
                sys.exit(0)

            if not args.confirm:
                count = int(client.command(f"SELECT count() FROM {validate_table_name(args.table)}"))
                print(f"\n⚠️  WARNING: This will delete {count} documents from '{args.table}'")
                confirm = input("Type 'yes' to confirm: ").strip().lower()
                if confirm != "yes":
                    print("❌ Cancelled")
                    sys.exit(0)

            client.command(f"DROP TABLE IF EXISTS {validate_table_name(args.table)}")
            print(f"✓ Table '{args.table}' cleared")

        elif args.action == "index":
            # Index documents
            if not args.source:
                print("❌ Error: --source required for indexing")
                print("\nUsage:")
                print("  python main.py --action index --source ./documents")
                sys.exit(1)

            from pathlib import Path

            source_path = Path(args.source)
            if not source_path.exists():
                print(f"❌ Error: Source folder not found: {args.source}")
                sys.exit(1)

            print("\n" + "=" * 60)
            print("📚 INDEXING CONFIGURATION")
            print("=" * 60)
            print(f"Source: {args.source}")
            print(f"Source type: {args.source_type}")
            print(f"Table: {args.table}")
            print(f"Batch size: {args.batch_size}")
            print(f"Lemmatization: {args.lemmatize}")
            print(f"Parallel processing: {not args.no_parallel}")
            print(f"Force recreate: {args.force_recreate}")
            print("=" * 60 + "\n")

            # Check if table exists and has data
            if table_exists(args.table) and not args.force_recreate:
                count = int(client.command(f"SELECT count() FROM {validate_table_name(args.table)}"))
                if count > 0:
                    print(f"ℹ️  Table '{args.table}' already contains {count} documents")
                    print("   New documents will be added (duplicates skipped)")
                    print("   Use --force-recreate to start fresh\n")

            store, nodes = initialize_clickhouse(
                documents_source=args.source,
                source_type=args.source_type,
                db_host=args.host,
                db_port=args.port,
                db_user=args.user,
                db_password=args.password,
                db_name=args.database,
                table_name=args.table,
                force_recreate=args.force_recreate,
                batch_size=args.batch_size,
            )

            if store and nodes:
                print_stats(store, nodes)
                print("✅ Indexing completed successfully")
            else:
                print("❌ Indexing failed (check logs above)")
                sys.exit(1)

        elif args.action == "stats":
            # Show statistics
            if not table_exists(args.table):
                print("\n" + "=" * 60)
                print(f"📋 Table '{args.table}' not found")
                print("=" * 60)
                print("\nThe table has not been created yet.")
                print("\nTo get started:")
                print("\n1. Index documents:")
                print("   python main.py --action index --source <folder>\n")
                print("2. Or initialize empty table:")
                print("   python main.py --action init\n")
                print("Available tables in database:")

                try:
                    result = client.query("SHOW TABLES")
                    tables = [row[0] for row in result.result_rows]
                    if tables:
                        for table in tables:
                            count = int(client.command(f"SELECT count() FROM {table}"))
                            print(f"  • {table} ({count} documents)")
                    else:
                        print("  (no tables found)")
                except Exception as e:
                    logging.debug("Failed to list tables: %s", e)

                print("=" * 60)
                sys.exit(0)

            store, nodes = load_existing_data(client, args.table)
            if store and nodes:
                print_stats(store, nodes)
            else:
                print(f"❌ Failed to load data from table '{args.table}'")
                sys.exit(1)

        elif args.action == "search":
            # Search mode
            if not table_exists(args.table):
                print(f"❌ Table '{args.table}' does not exist")
                print("   Run: python main.py --action index --source <folder>")
                sys.exit(1)

            store, nodes = load_existing_data(client, args.table)
            if not store or not nodes:
                print(f"❌ Table '{args.table}' is empty")
                sys.exit(1)

            embed_model = FridaEmbedding()
            Settings.embed_model = embed_model

            if args.query:
                # Single query mode
                print(f"\n🔎 Searching: {args.query}\n")
                query_emb = embed_model.get_text_embedding(args.query)
                results = store.search_by_cosine(query_emb, limit=args.limit)

                if not results:
                    print("No results found")
                else:
                    for i, result in enumerate(results, 1):
                        dist = result.get("dist", 0)
                        text = result.get("text", "")[:200]
                        print(f"{i}. [Distance: {dist:.4f}]")
                        print(f"   {text}...\n")
            else:
                # Interactive mode
                interactive_search(store, embed_model)

        elif args.action == "benchmark":
            # Run benchmarks
            if not table_exists(args.table):
                print(f"❌ Table '{args.table}' does not exist")
                sys.exit(1)

            store, nodes = load_existing_data(client, args.table)
            if not store or not nodes:
                print(f"❌ Table '{args.table}' is empty")
                sys.exit(1)

            embed_model = FridaEmbedding()
            Settings.embed_model = embed_model

            benchmark_search(store, embed_model, args.num_queries)

        elif args.action == "export":
            # Export to JSON
            if not table_exists(args.table):
                print(f"❌ Table '{args.table}' does not exist")
                sys.exit(1)

            store, nodes = load_existing_data(client, args.table)
            if not store or not nodes:
                print(f"❌ Table '{args.table}' is empty")
                sys.exit(1)

            export_to_json(store, nodes, args.output)

        # Cleanup
        client.close()

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.exception("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    # run_scraper_separate_files()
    main()
