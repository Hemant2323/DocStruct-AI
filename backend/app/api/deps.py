"""Shared FastAPI dependencies.

Dependencies are how the HTTP layer assembles the objects a route needs — a database
session, a repository, a service — without those routes constructing them by hand. It
also makes them trivially replaceable in tests via ``app.dependency_overrides``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.embeddings.base import Embedder
from app.ai.embeddings.gemini import GeminiEmbedder
from app.ai.providers.base import LLMProvider
from app.ai.providers.claude import ClaudeProvider
from app.ai.providers.gemini import GeminiProvider
from app.ai.providers.nvidia import NVIDIAProvider
from app.core.config import Settings, get_settings
from app.core.database import get_db_session, get_session_factory
from app.core.exceptions import ConfigurationError
from app.repositories.ai_result_repo import AIResultRepository
from app.repositories.block_repo import ContentBlockRepository
from app.repositories.chunk_embedding_repo import ChunkEmbeddingRepository
from app.repositories.chunk_repo import ChunkRepository
from app.repositories.conversion_repo import ConversionRepository
from app.repositories.document_repo import DocumentRepository
from app.retrieval.base import Retriever
from app.retrieval.cosine import CosineSimilarityRetriever
from app.services.analysis_service import AnalysisService
from app.services.conversion_service import ConversionService
from app.services.document_service import DocumentService
from app.services.export_service import ExportService
from app.services.ingestion_service import IngestionService
from app.services.job_service import JobService
from app.services.qa_service import QAService
from app.storage.base import FileStorage
from app.storage.local import LocalFileStorage
from app.storage.supabase import SupabaseFileStorage

# Milestone 1 has no authentication, so every request acts as this owner. The seam is
# real: routes and services already take an owner_id and every query filters on it, so
# adding authentication later means changing this one function to read the authenticated
# user instead of returning a constant.
DEFAULT_OWNER_ID = "local-user"


def get_current_owner_id() -> str:
    """Return the owner making this request.

    Replaced in a later milestone by real authentication. Until then every request is
    attributed to a single local user.
    """
    return DEFAULT_OWNER_ID


#: Process-wide storage backend, held only when the selected one owns an HTTP connection
#: pool. See `get_file_storage` for why the local backend is deliberately not cached.
_file_storage: FileStorage | None = None


def _build_file_storage(settings: Settings) -> FileStorage:
    """Construct the backend named by ``settings.storage_backend``.

    Kept separate from :func:`get_file_storage` for the same reason as
    :func:`_build_provider`: the branching — the one part that changes when a backend is
    added — is a small, direct function a test can call without FastAPI's dependency
    machinery, a request, or a network.

    Raises:
        ConfigurationError: ``supabase`` was selected without the credentials it needs.
            Raised here rather than falling back to local storage, because a silent
            fallback would write uploads to a disk the operator believes is unused and
            that a free host discards on the next deploy — the documents would be gone
            and nothing would have said so.
    """
    if settings.storage_backend == "supabase":
        if not settings.supabase_url or not settings.supabase_service_key:
            raise ConfigurationError(
                "STORAGE_BACKEND is 'supabase' but SUPABASE_URL and SUPABASE_SERVICE_KEY "
                "are not both set."
            )
        return SupabaseFileStorage(
            base_url=settings.supabase_url,
            service_key=settings.supabase_service_key,
            bucket=settings.supabase_storage_bucket,
        )

    # "local" is the default and the fallback for any value the validator already
    # guarantees is one of the two known names — this branch cannot see anything else.
    return LocalFileStorage(settings.storage_dir)


def get_file_storage(settings: Annotated[Settings, Depends(get_settings)]) -> FileStorage:
    """Return the configured file storage backend.

    Typed as the interface, not the implementation, so the services and routes that
    depend on it cannot tell which backend is in use.

    **Only the Supabase backend is cached.** `_provider` and `_embedder` are held
    process-wide because they own HTTP connection pools that should not be rebuilt per
    request, and `SupabaseFileStorage` is the same. `LocalFileStorage` is not: it holds a
    resolved path and nothing else, so caching it would buy nothing and would pin the
    first request's ``storage_dir`` for the life of the process — which is exactly what
    the test suite varies per test.
    """
    if settings.storage_backend != "supabase":
        return _build_file_storage(settings)

    global _file_storage
    if _file_storage is None:
        _file_storage = _build_file_storage(settings)
    return _file_storage


def reset_file_storage() -> None:
    """Drop the cached storage backend.

    Used by tests, which change settings between cases and would otherwise get a backend
    built from a previous test's configuration. Mirrors `reset_llm_provider` and
    `reset_embedder`.
    """
    global _file_storage
    _file_storage = None


def get_document_repository(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> DocumentRepository:
    """Return a document repository bound to this request's session."""
    return DocumentRepository(session)


def get_block_repository(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ContentBlockRepository:
    """Return a content block repository bound to this request's session."""
    return ContentBlockRepository(session)


def get_chunk_repository(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ChunkRepository:
    """Return a chunk repository bound to this request's session."""
    return ChunkRepository(session)


def get_document_service(
    repository: Annotated[DocumentRepository, Depends(get_document_repository)],
    storage: Annotated[FileStorage, Depends(get_file_storage)],
    settings: Annotated[Settings, Depends(get_settings)],
    block_repository: Annotated[ContentBlockRepository, Depends(get_block_repository)],
    chunk_repository: Annotated[ChunkRepository, Depends(get_chunk_repository)],
) -> DocumentService:
    """Return a fully wired document service."""
    return DocumentService(
        repository,
        storage,
        settings,
        block_repository=block_repository,
        chunk_repository=chunk_repository,
    )


def get_ai_result_repository(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> AIResultRepository:
    """Return an AI result repository bound to this request's session."""
    return AIResultRepository(session)


def get_chunk_embedding_repository(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ChunkEmbeddingRepository:
    """Return a chunk embedding repository bound to this request's session."""
    return ChunkEmbeddingRepository(session)


#: Process-wide provider. Held here rather than built per request because the client
#: owns an HTTP connection pool; rebuilding it on every call would discard those
#: connections and add latency to every AI request.
#:
#: Deliberately NOT `functools.lru_cache` keyed on Settings — a Pydantic model is not
#: hashable, so that fails at runtime with `unhashable type: 'Settings'`.
_provider: LLMProvider | None = None


def _build_provider(settings: Settings) -> LLMProvider:
    """Construct the provider named by ``settings.ai_provider``.

    Kept separate from :func:`get_llm_provider` so the branching itself — the one part
    that changes when a provider is added — is a small, direct function that a test can
    call without going through FastAPI's dependency machinery.
    """
    if settings.ai_provider == "nvidia":
        return NVIDIAProvider(
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
            model=settings.nvidia_model,
            timeout_seconds=settings.ai_timeout_seconds,
            max_retries=settings.ai_max_retries,
            max_output_tokens=settings.ai_max_output_tokens,
        )

    if settings.ai_provider == "gemini":
        return GeminiProvider(
            api_key=settings.gemini_api_key,
            base_url=settings.gemini_base_url,
            model=settings.gemini_model,
            timeout_seconds=settings.ai_timeout_seconds,
            max_retries=settings.ai_max_retries,
            max_output_tokens=settings.ai_max_output_tokens,
        )

    # "claude" is the default and the fallback for any value the validator already
    # guarantees is one of the three known names — this branch cannot see anything else.
    return ClaudeProvider(settings)


def get_llm_provider(settings: Annotated[Settings, Depends(get_settings)]) -> LLMProvider:
    """Return the configured language model provider.

    Typed as the interface, not the implementation, so swapping providers later touches
    this function and nothing else. Which concrete class gets built is controlled by
    ``AI_PROVIDER`` ("claude", the default, "nvidia", or "gemini") and decided once, the
    first time an AI feature is used.
    """
    global _provider
    if _provider is None:
        _provider = _build_provider(settings)
    return _provider


def reset_llm_provider() -> None:
    """Drop the cached provider.

    Used by tests, which change settings between cases and would otherwise get a
    provider built from a previous test's configuration.
    """
    global _provider
    _provider = None


def get_analysis_service(
    provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
    documents: Annotated[DocumentRepository, Depends(get_document_repository)],
    chunks: Annotated[ChunkRepository, Depends(get_chunk_repository)],
    blocks: Annotated[ContentBlockRepository, Depends(get_block_repository)],
    results: Annotated[AIResultRepository, Depends(get_ai_result_repository)],
) -> AnalysisService:
    """Return a fully wired analysis service."""
    return AnalysisService(
        provider=provider,
        settings=settings,
        documents=documents,
        chunks=chunks,
        blocks=blocks,
        results=results,
    )


def get_export_service(
    provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    documents: Annotated[DocumentRepository, Depends(get_document_repository)],
    results: Annotated[AIResultRepository, Depends(get_ai_result_repository)],
) -> ExportService:
    """Return a fully wired export service.

    Depends on the provider only to read its ``model_name`` when recomputing a cache
    fingerprint — export never calls the provider, matching `ExportService`'s own
    read-only contract.
    """
    return ExportService(provider=provider, documents=documents, results=results)


#: Process-wide embedder, held for the same reason as `_provider`: it owns an HTTP
#: connection pool that should not be rebuilt on every call.
_embedder: Embedder | None = None


def get_embedder(settings: Annotated[Settings, Depends(get_settings)]) -> Embedder:
    """Return the configured embedding provider.

    Only one embedding implementation exists (Gemini's) — unlike `get_llm_provider`,
    which branches across three. Typed as the interface anyway, so adding a second
    embedding provider later touches only this function.

    Uses `gemini_embedding_model`, not `gemini_model` — the latter is a generation/chat
    model (GeminiProvider's), which is not a valid model for the embeddings endpoint.
    """
    global _embedder
    if _embedder is None:
        _embedder = GeminiEmbedder(
            api_key=settings.gemini_api_key,
            base_url=settings.gemini_base_url,
            model=settings.gemini_embedding_model,
            timeout_seconds=settings.ai_timeout_seconds,
            max_retries=settings.ai_max_retries,
        )
    return _embedder


def reset_embedder() -> None:
    """Drop the cached embedder.

    Used by tests, which change settings between cases and would otherwise get an
    embedder built from a previous test's configuration. Mirrors `reset_llm_provider`.
    """
    global _embedder
    _embedder = None


def get_retriever(
    embedder: Annotated[Embedder, Depends(get_embedder)],
    chunks: Annotated[ChunkRepository, Depends(get_chunk_repository)],
    chunk_embeddings: Annotated[ChunkEmbeddingRepository, Depends(get_chunk_embedding_repository)],
) -> Retriever:
    """Return the configured retriever.

    Typed as the interface, not the implementation, so swapping the retrieval strategy
    later touches this function and nothing else — the same reasoning as
    `get_llm_provider`.
    """
    return CosineSimilarityRetriever(embedder=embedder, chunks=chunks, embeddings=chunk_embeddings)


def get_qa_service(
    retriever: Annotated[Retriever, Depends(get_retriever)],
    provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
    documents: Annotated[DocumentRepository, Depends(get_document_repository)],
    blocks: Annotated[ContentBlockRepository, Depends(get_block_repository)],
) -> QAService:
    """Return a fully wired Q&A service."""
    return QAService(
        retriever=retriever,
        provider=provider,
        settings=settings,
        documents=documents,
        blocks=blocks,
    )


def get_ingestion_service(
    storage: Annotated[FileStorage, Depends(get_file_storage)],
    settings: Annotated[Settings, Depends(get_settings)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> IngestionService:
    """Return the ingestion pipeline.

    Built with the session **factory**, not the request's session. Ingestion runs in a
    background task after the response has been sent, by which point the request-scoped
    session is already committed and closed — using it would fail at runtime.

    The embedder is passed through so ingestion can generate chunk embeddings after a
    document is ready — see `IngestionService._generate_embeddings`. Its own
    `is_configured` check means this is safe to wire unconditionally: an unconfigured
    embedder simply results in ingestion skipping that step, exactly as before this was
    wired in.
    """
    return IngestionService(get_session_factory(), storage, settings, embedder)


def get_conversion_repository(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ConversionRepository:
    """Return a conversion repository bound to this request's session."""
    return ConversionRepository(session)


def get_conversion_service(
    analysis: Annotated[AnalysisService, Depends(get_analysis_service)],
    provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ConversionService:
    """Return a fully wired conversion service.

    Built on top of `AnalysisService` rather than beside it: conversion reads its source
    records through the extraction pipeline that already exists, and does not have one
    of its own.
    """
    return ConversionService(analysis=analysis, provider=provider, settings=settings)


def get_job_service(request: Request) -> JobService:
    """Return the application-wide background job scheduler.

    Stored on application state rather than created per request, so its concurrency limit
    applies across all requests rather than being reset by each one.
    """
    job_service = getattr(request.app.state, "job_service", None)
    if job_service is None:  # pragma: no cover - guaranteed by the lifespan handler
        raise RuntimeError("JobService was not initialised during application startup")
    return job_service


# Aliases keep route signatures readable.
SettingsDep = Annotated[Settings, Depends(get_settings)]
OwnerIdDep = Annotated[str, Depends(get_current_owner_id)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
JobServiceDep = Annotated[JobService, Depends(get_job_service)]
AnalysisServiceDep = Annotated[AnalysisService, Depends(get_analysis_service)]
LLMProviderDep = Annotated[LLMProvider, Depends(get_llm_provider)]
QAServiceDep = Annotated[QAService, Depends(get_qa_service)]
ExportServiceDep = Annotated[ExportService, Depends(get_export_service)]
ConversionServiceDep = Annotated[ConversionService, Depends(get_conversion_service)]
ConversionRepositoryDep = Annotated[ConversionRepository, Depends(get_conversion_repository)]
