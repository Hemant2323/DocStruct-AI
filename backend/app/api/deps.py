"""Shared FastAPI dependencies.

Dependencies are how the HTTP layer assembles the objects a route needs — a database
session, a repository, a service — without those routes constructing them by hand. It
also makes them trivially replaceable in tests via ``app.dependency_overrides``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.database import get_db_session
from app.repositories.document_repo import DocumentRepository
from app.services.document_service import DocumentService
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


def get_document_service(
    repository: Annotated[DocumentRepository, Depends(get_document_repository)],
    storage: Annotated[FileStorage, Depends(get_file_storage)],
) -> DocumentService:
    """Return a fully wired document service."""
    return DocumentService(repository, storage)


# Aliases keep route signatures readable.
SettingsDep = Annotated[Settings, Depends(get_settings)]
OwnerIdDep = Annotated[str, Depends(get_current_owner_id)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
