"""Shared FastAPI dependencies.

Dependencies are how the HTTP layer assembles the objects a route needs — a database
session, a repository, a service — without those routes constructing them by hand. It
also makes them trivially replaceable in tests via ``app.dependency_overrides``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.providers.base import LLMProvider
from app.ai.providers.claude import ClaudeProvider
from app.core.config import Settings, get_settings
from app.core.database import get_db_session, get_session_factory
from app.repositories.ai_result_repo import AIResultRepository
from app.repositories.block_repo import ContentBlockRepository
from app.repositories.chunk_repo import ChunkRepository
from app.repositories.document_repo import DocumentRepository
from app.services.analysis_service import AnalysisService
from app.services.document_service import DocumentService
from app.services.ingestion_service import IngestionService
from app.services.job_service import JobService
from app.storage.base import FileStorage
from app.storage.local import LocalFileStorage

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


def get_file_storage(settings: Annotated[Settings, Depends(get_settings)]) -> FileStorage:
    """Return the configured file storage backend."""
    return LocalFileStorage(settings.storage_dir)


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


#: Process-wide provider. Held here rather than built per request because the client
#: owns an HTTP connection pool; rebuilding it on every call would discard those
#: connections and add latency to every AI request.
#:
#: Deliberately NOT `functools.lru_cache` keyed on Settings — a Pydantic model is not
#: hashable, so that fails at runtime with `unhashable type: 'Settings'`.
_provider: LLMProvider | None = None


def get_llm_provider(settings: Annotated[Settings, Depends(get_settings)]) -> LLMProvider:
    """Return the configured language model provider.

    Typed as the interface, not the implementation, so swapping providers later touches
    this function and nothing else.
    """
    global _provider
    if _provider is None:
        _provider = ClaudeProvider(settings)
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


def get_ingestion_service(
    storage: Annotated[FileStorage, Depends(get_file_storage)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> IngestionService:
    """Return the ingestion pipeline.

    Built with the session **factory**, not the request's session. Ingestion runs in a
    background task after the response has been sent, by which point the request-scoped
    session is already committed and closed — using it would fail at runtime.
    """
    return IngestionService(get_session_factory(), storage, settings)


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
