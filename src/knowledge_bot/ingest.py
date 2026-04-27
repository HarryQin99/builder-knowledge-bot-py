"""Phase 2 ingestion CLI.

Two-tier idempotency:
  1. md5 the PDF bytes; compare to corpus_metadata. If match -> exit early.
  2. On mismatch: chunk, compute SHA-256 chunk IDs, SELECT existing IDs,
     embed and INSERT only the new chunks. UPSERT corpus_metadata.

Run: uv run python -m knowledge_bot.ingest
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import psycopg
import pymupdf
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from knowledge_bot.config import Settings, get_settings


CORPUS_METADATA_DDL = """
CREATE TABLE IF NOT EXISTS corpus_metadata (
    corpus_id    TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    chunk_count  INT  NOT NULL,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


def md5_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def deterministic_chunk_id(corpus_filename: str, page: int, idx: int, text: str) -> str:
    raw = f"{corpus_filename}|{page}|{idx}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_documents(corpus_path: Path) -> list[Document]:
    doc = pymupdf.open(str(corpus_path))
    try:
        return [
            Document(
                text=page.get_text(),
                metadata={"page_number": i + 1, "source": corpus_path.name},
            )
            for i, page in enumerate(doc)
        ]
    finally:
        doc.close()


def existing_corpus_hash(conn: psycopg.Connection, corpus_id: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute("SELECT content_hash FROM corpus_metadata WHERE corpus_id = %s", (corpus_id,))
        row = cur.fetchone()
        return row[0] if row else None


def upsert_corpus_metadata(
    conn: psycopg.Connection, corpus_id: str, content_hash: str, chunk_count: int
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO corpus_metadata (corpus_id, content_hash, chunk_count, ingested_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (corpus_id) DO UPDATE
              SET content_hash = EXCLUDED.content_hash,
                  chunk_count = EXCLUDED.chunk_count,
                  ingested_at = EXCLUDED.ingested_at
            """,
            (corpus_id, content_hash, chunk_count),
        )
    conn.commit()


def existing_chunk_ids(conn: psycopg.Connection, ids: list[str], table: str) -> set[str]:
    if not ids:
        return set()
    try:
        with conn.cursor() as cur:
            cur.execute(f'SELECT id FROM "{table}" WHERE id = ANY(%s)', (ids,))
            return {row[0] for row in cur.fetchall()}
    except psycopg.errors.UndefinedTable:
        conn.rollback()
        return set()


def build_pg_store(settings: Settings) -> PGVectorStore:
    return PGVectorStore.from_params(
        database=settings.postgres_db,
        host=settings.postgres_host,
        password=settings.postgres_password,
        port=settings.postgres_port,
        user=settings.postgres_user,
        table_name=settings.vector_table_name,
        embed_dim=settings.embedding_dim,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )


def main() -> None:
    settings = get_settings()
    corpus = settings.corpus_path

    if not corpus.exists():
        sys.exit(f"Corpus not found at {corpus}. Place the NCC PDF there first.")

    print(f"Ingest starting. Corpus: {corpus}")
    current_hash = md5_file(corpus)

    conn_kwargs = dict(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        dbname=settings.postgres_db,
    )

    with psycopg.connect(**conn_kwargs) as conn:
        with conn.cursor() as cur:
            cur.execute(CORPUS_METADATA_DDL)
        conn.commit()

        stored = existing_corpus_hash(conn, settings.corpus_id)
        if stored == current_hash:
            print(f"Corpus unchanged (md5={current_hash[:12]}...). Skipping ingestion.")
            return

        print("Corpus changed or first ingest. Running full pipeline...")

        t0 = time.perf_counter()
        documents = load_documents(corpus)
        print(f"  PDF loaded: {len(documents)} pages in {time.perf_counter() - t0:.2f}s")

        t1 = time.perf_counter()
        splitter = SentenceSplitter(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        nodes = splitter.get_nodes_from_documents(documents)
        print(f"  Chunked: {len(nodes)} chunks in {time.perf_counter() - t1:.2f}s")

        for i, node in enumerate(nodes):
            page = node.metadata.get("page_number", 0)
            node.id_ = deterministic_chunk_id(corpus.name, page, i, node.text)

        ids = [n.id_ for n in nodes]
        existing = existing_chunk_ids(conn, ids, settings.vector_table_full)
        new_nodes = [n for n in nodes if n.id_ not in existing]
        print(f"  In store: {len(existing)}; new to embed: {len(new_nodes)}")

        if new_nodes:
            t2 = time.perf_counter()
            pg_store = build_pg_store(settings)
            embed_model = OpenAIEmbedding(
                model=settings.embedding_model, api_key=settings.openai_api_key
            )
            index = VectorStoreIndex.from_vector_store(
                vector_store=pg_store, embed_model=embed_model
            )
            index.insert_nodes(new_nodes)
            print(
                f"  Embedded + inserted {len(new_nodes)} chunks "
                f"in {time.perf_counter() - t2:.2f}s"
            )
        else:
            print("  No new chunks to embed; updating metadata only.")

        upsert_corpus_metadata(conn, settings.corpus_id, current_hash, len(nodes))
        print(f"Ingest complete. Total chunks: {len(nodes)}; hash: {current_hash[:12]}...")


if __name__ == "__main__":
    main()
