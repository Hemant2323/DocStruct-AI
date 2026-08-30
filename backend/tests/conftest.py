"""Shared pytest fixtures.

Every test runs against its own throwaway SQLite database in a temporary directory, so
tests never share state and can run in any order. The application's global engine is
disposed and rebuilt around each test, which is why ``get_settings.cache_clear()`` and
``dispose_engine()`` appear in the teardown.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_llm_provider, get_retriever, reset_llm_provider
from app.core.config import Settings, get_settings
from app.core.database import dispose_engine, get_engine, get_session_factory
from app.main import create_app
from app.models.db import Base
from app.repositories.document_repo import DocumentRepository
from app.services.job_service import JobService
from app.storage.local import LocalFileStorage
from tests.fixtures.fake_provider import FakeLLMProvider
from tests.fixtures.fake_retriever import FakeRetriever


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Application settings pointed at a temporary database and storage directory.

    Environment variables are set rather than a ``Settings`` object constructed directly,
    so this exercises the same configuration-loading path the real application uses.

    Every AI-related variable is pinned explicitly, not just ``ANTHROPIC_API_KEY``: without
    that, a value present only in the developer's own ``backend/.env`` (a non-default
    ``AI_PROVIDER``, a real key for another provider, a non-default ``ANTHROPIC_MODEL``)
    leaks into tests that assume the code's plain defaults, making them pass or fail
    depending on whose machine they run on rather than on the code itself.
    """
    db_path = tmp_path / "test.db"
    storage_path = tmp_path / "storage"
    storage_path.mkdir()

    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("STORAGE_DIR", str(storage_path))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("AI_PROVIDER", "claude")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-5")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "")

    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
async def _database(settings: Settings) -> AsyncGenerator[None, None]:
    """Create the schema in the temporary database, then tear the engine down.

    Tables are created from ``Base.metadata`` rather than by running migrations, because
    that is fast and every test needs it. Migrations get their own dedicated test
    (``tests/integration/test_migrations.py``) that verifies they stay in sync with the
    models — so skipping them here does not leave them unverified.
    """
    engine = get_engine()
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    yield

    await dispose_engine()
    get_settings.cache_clear()


@pytest.fixture
async def db_session(_database: None) -> AsyncGenerator[AsyncSession, None]:
    """A database session for tests that exercise repositories directly."""
    factory = get_session_factory()
    async with factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
def document_repository(db_session: AsyncSession) -> DocumentRepository:
    """A repository bound to the test session."""
    return DocumentRepository(db_session)


@pytest.fixture
def file_storage(settings: Settings) -> LocalFileStorage:
    """Local file storage rooted in the test's temporary directory."""
    return LocalFileStorage(settings.storage_dir)


@pytest.fixture(autouse=True)
def _reset_provider() -> Generator[None, None, None]:
    """Drop the cached LLM provider around every test.

    The provider is a process-wide singleton (it owns a connection pool), so without
    this a test would inherit one built from a previous test's settings.
    """
    reset_llm_provider()
    yield
    reset_llm_provider()


@pytest.fixture
def app(settings: Settings, _database: None) -> FastAPI:
    """The FastAPI application wired to the test database."""
    return create_app(settings)


@pytest.fixture
def fake_provider() -> FakeLLMProvider:
    """A stand-in language model. No test ever calls the real API."""
    return FakeLLMProvider()


@pytest.fixture
def ai_app(app: FastAPI, fake_provider: FakeLLMProvider) -> FastAPI:
    """The application with the language model replaced by a fake."""
    app.dependency_overrides[get_llm_provider] = lambda: fake_provider
    return app


@pytest.fixture
async def ai_client(ai_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """An HTTP client whose AI calls go to the fake provider."""
    transport = ASGITransport(app=ai_app)
    async with (
        ai_app.router.lifespan_context(ai_app),
        AsyncClient(transport=transport, base_url="http://testserver") as http_client,
    ):
        yield http_client


@pytest.fixture
def fake_retriever() -> FakeRetriever:
    """A stand-in retriever. No test here computes real embeddings or similarity."""
    return FakeRetriever()


@pytest.fixture
def qa_app(ai_app: FastAPI, fake_retriever: FakeRetriever) -> FastAPI:
    """The application with both the language model and the retriever replaced by fakes."""
    ai_app.dependency_overrides[get_retriever] = lambda: fake_retriever
    return ai_app


@pytest.fixture
async def qa_client(qa_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """An HTTP client whose AI and retrieval calls both go to fakes."""
    transport = ASGITransport(app=qa_app)
    async with (
        qa_app.router.lifespan_context(qa_app),
        AsyncClient(transport=transport, base_url="http://testserver") as http_client,
    ):
        yield http_client


@pytest.fixture
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """An HTTP client that talks to the app in-process, with startup/shutdown run.

    ``ASGITransport`` dispatches straight to the application, so no port is bound and no
    real network is involved — the tests stay fast and cannot collide with a running dev
    server.

    The lifespan context is entered explicitly because ``ASGITransport`` does not run it.
    Without that, ``app.state.job_service`` would never be created and every upload would
    fail — and the startup reaper would go untested.
    """
    transport = ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=transport, base_url="http://testserver") as http_client,
    ):
        yield http_client


@pytest.fixture
def job_service(app: FastAPI) -> JobService:
    """The application's background job scheduler.

    Tests await ``wait_for_all()`` on this to let ingestion finish before asserting,
    rather than sleeping and hoping.
    """
    return app.state.job_service  # type: ignore[no-any-return]


@pytest.fixture
def api_prefix(settings: Settings) -> str:
    """The configured API prefix, so tests do not hardcode '/api'."""
    return settings.api_prefix
