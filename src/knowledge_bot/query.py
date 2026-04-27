"""Phase 2 query engine factory.

Builds the LlamaIndex RetrieverQueryEngine once at startup with explicit
named components: PGVectorStore + OpenAIEmbedding + Anthropic LLM.
Returned alongside a TokenCountingHandler so the AskService can read
per-query token usage.
"""
from __future__ import annotations

from llama_index.core import VectorStoreIndex
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.postgres import PGVectorStore

from knowledge_bot.config import get_settings


def build_query_engine() -> tuple[RetrieverQueryEngine, TokenCountingHandler]:
    """Wire the RAG query path. Returns (engine, token_counter)."""
    settings = get_settings()

    embed_model = OpenAIEmbedding(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    llm = Anthropic(
        model=settings.chat_model,
        api_key=settings.anthropic_api_key,
    )

    pg_store = PGVectorStore.from_params(
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

    token_counter = TokenCountingHandler()
    callback_manager = CallbackManager([token_counter])

    index = VectorStoreIndex.from_vector_store(
        vector_store=pg_store,
        embed_model=embed_model,
        callback_manager=callback_manager,
    )
    retriever = index.as_retriever(similarity_top_k=settings.top_k)

    engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        callback_manager=callback_manager,
    )
    return engine, token_counter
