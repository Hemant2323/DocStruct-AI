"""Tests for background processing, the startup reaper, and job scheduling.

These target the failure modes that only appear once work runs *outside* a request: a
closed session, a crash mid-parse, and unbounded concurrency.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.database import get_session_factory
from app.core.exceptions import NotFoundError, ScannedDocumentError
from app.models.domain.document import ProcessingStatus, SupportedFileType
from app.repositories.block_repo import ContentBlockRepository
from app.repositories.chunk_repo import ChunkRepository
from app.repositories.document_repo import DocumentRepository
from app.services.ingestion_service import (
    INTERRUPTED_FAILURE_CODE,
    IngestionService,
    reap_interrupted_documents,
)
from app.services.job_service import JobService
from app.storage.local import LocalFileStorage
from tests.fixtures.builders import build_txt, read_pdf_fixture

OWNER = "local-user"


class TestIngestionServiceUsesItsOwnSession:
    async def test_processes_a_document_end_to_end(
        self, settings: Settings, db_session: AsyncSession, file_storage: LocalFileStorage
    ) -> None:
        """The real risk this guards: the request session is closed by now.

        IngestionService is built with a session *factory* precisely so it can open its
        own. A mocked test would never notice if that were wrong; this one would fail.
        """
        repository = DocumentRepository(db_session)
        document = await repository.create(
            owner_id=OWNER,
            original_filename="notes.txt",
            file_type=SupportedFileType.TXT,
            size_bytes=100,
        )
        await db_session.commit()
        await file_storage.save(document.id, build_txt("First.\n\nSecond paragraph."))

        service = IngestionService(get_session_factory(), file_storage, settings)
        result = await service.process_document(document.id, owner_id=OWNER)

        assert result.block_count == 2
        assert result.chunk_count >= 1

        # Read back through a fresh session to confirm it was really committed.
        async with get_session_factory()() as session:
            stored = await DocumentRepository(session).get(document.id, owner_id=OWNER)
            blocks = await ContentBlockRepository(session).count_for_document(document.id)
            chunks = await ChunkRepository(session).count_for_document(document.id)

        assert stored is not None
        assert stored.status is ProcessingStatus.READY
        assert blocks == 2
        assert chunks >= 1

    async def test_records_page_count_on_success(
        self, settings: Settings, db_session: AsyncSession, file_storage: LocalFileStorage
    ) -> None:
        repository = DocumentRepository(db_session)
        document = await repository.create(
            owner_id=OWNER,
            original_filename="report.pdf",
            file_type=SupportedFileType.PDF,
            size_bytes=1000,
        )
        await db_session.commit()
        await file_storage.save(document.id, read_pdf_fixture("simple.pdf"))

        service = IngestionService(get_session_factory(), file_storage, settings)
        result = await service.process_document(document.id, owner_id=OWNER)

        assert result.page_count == 2

    async def test_failure_leaves_a_terminal_state_with_a_reason(
        self, settings: Settings, db_session: AsyncSession, file_storage: LocalFileStorage
    ) -> None:
        repository = DocumentRepository(db_session)
        document = await repository.create(
            owner_id=OWNER,
            original_filename="scan.pdf",
            file_type=SupportedFileType.PDF,
            size_bytes=1000,
        )
        await db_session.commit()
        await file_storage.save(document.id, read_pdf_fixture("scanned.pdf"))

        service = IngestionService(get_session_factory(), file_storage, settings)

        with pytest.raises(ScannedDocumentError):
            await service.process_document(document.id, owner_id=OWNER)

        async with get_session_factory()() as session:
            stored = await DocumentRepository(session).get(document.id, owner_id=OWNER)

        assert stored is not None
        assert stored.status is ProcessingStatus.FAILED
        assert stored.failure_code == "requires_ocr"
        assert stored.error_message

    async def test_missing_stored_file_fails_cleanly(
        self, settings: Settings, db_session: AsyncSession, file_storage: LocalFileStorage
    ) -> None:
        """A row whose bytes are gone must fail with a reason, not hang in PROCESSING."""
        repository = DocumentRepository(db_session)
        document = await repository.create(
            owner_id=OWNER,
            original_filename="ghost.txt",
            file_type=SupportedFileType.TXT,
            size_bytes=10,
        )
        await db_session.commit()
        # Deliberately no file saved.

        service = IngestionService(get_session_factory(), file_storage, settings)

        # Storage reports a missing file as NotFoundError; the point of the test is the
        # document's end state, but asserting the specific type keeps it honest.
        with pytest.raises(NotFoundError):
            await service.process_document(document.id, owner_id=OWNER)

        async with get_session_factory()() as session:
            stored = await DocumentRepository(session).get(document.id, owner_id=OWNER)

        assert stored is not None
        assert stored.status is ProcessingStatus.FAILED
        assert stored.error_message

    async def test_reprocessing_replaces_rather_than_duplicates(
        self, settings: Settings, db_session: AsyncSession, file_storage: LocalFileStorage
    ) -> None:
        """Running the pipeline twice must not leave two generations of content."""
        repository = DocumentRepository(db_session)
        document = await repository.create(
            owner_id=OWNER,
            original_filename="a.txt",
            file_type=SupportedFileType.TXT,
            size_bytes=50,
        )
        await db_session.commit()
        await file_storage.save(document.id, build_txt("One.\n\nTwo.\n\nThree."))

        service = IngestionService(get_session_factory(), file_storage, settings)
        first = await service.process_document(document.id, owner_id=OWNER)
        second = await service.process_document(document.id, owner_id=OWNER)

        assert first.block_count == second.block_count

        async with get_session_factory()() as session:
            total = await ContentBlockRepository(session).count_for_document(document.id)

        assert total == second.block_count


class TestStartupReaper:
    async def test_fails_documents_stranded_in_processing(self, db_session: AsyncSession) -> None:
        """A document only sits in PROCESSING while a task works on it.

        If one is in that state at startup, the previous process died mid-parse and
        nothing will ever finish it — without this it would spin in the UI forever.
        """
        repository = DocumentRepository(db_session)
        document = await repository.create(
            owner_id=OWNER,
            original_filename="stuck.pdf",
            file_type=SupportedFileType.PDF,
            size_bytes=100,
        )
        await repository.update_status(
            document.id, owner_id=OWNER, status=ProcessingStatus.PROCESSING
        )
        await db_session.commit()

        reaped = await reap_interrupted_documents(get_session_factory())

        assert reaped == 1

        async with get_session_factory()() as session:
            stored = await DocumentRepository(session).get(document.id, owner_id=OWNER)

        assert stored is not None
        assert stored.status is ProcessingStatus.FAILED
        assert stored.failure_code == INTERRUPTED_FAILURE_CODE
        assert "restart" in stored.error_message.lower()

    async def test_leaves_other_statuses_alone(self, db_session: AsyncSession) -> None:
        repository = DocumentRepository(db_session)
        for status in (
            ProcessingStatus.PENDING,
            ProcessingStatus.READY,
        ):
            document = await repository.create(
                owner_id=OWNER,
                original_filename=f"{status.value}.txt",
                file_type=SupportedFileType.TXT,
                size_bytes=10,
            )
            await repository.update_status(document.id, owner_id=OWNER, status=status)
        await db_session.commit()

        assert await reap_interrupted_documents(get_session_factory()) == 0

    async def test_is_safe_when_nothing_is_stranded(self, _database: None) -> None:
        assert await reap_interrupted_documents(get_session_factory()) == 0


class TestJobService:
    async def test_runs_scheduled_work(self) -> None:
        ran = asyncio.Event()

        async def work() -> None:
            ran.set()

        jobs = JobService(max_concurrent=2)
        jobs.schedule("test", work)
        await jobs.wait_for_all(timeout=5.0)

        assert ran.is_set()

    async def test_a_failing_job_does_not_escape(self) -> None:
        """A background failure must be logged, not crash the event loop."""

        async def work() -> None:
            raise RuntimeError("job blew up")

        jobs = JobService(max_concurrent=2)
        jobs.schedule("failing", work)

        await jobs.wait_for_all(timeout=5.0)  # must not raise

        assert jobs.active_count == 0

    async def test_concurrency_is_bounded(self) -> None:
        """Without a bound, ten large uploads would be parsed simultaneously."""
        concurrent = 0
        peak = 0
        release = asyncio.Event()

        async def work() -> None:
            nonlocal concurrent, peak
            concurrent += 1
            peak = max(peak, concurrent)
            await release.wait()
            concurrent -= 1

        jobs = JobService(max_concurrent=2)
        for index in range(6):
            jobs.schedule(f"job-{index}", work)

        await asyncio.sleep(0.05)  # let the permitted jobs start
        release.set()
        await jobs.wait_for_all(timeout=5.0)

        assert peak <= 2

    async def test_tasks_are_tracked_so_they_cannot_be_garbage_collected(self) -> None:
        """asyncio holds only weak references; a lost task would vanish mid-flight."""
        started = asyncio.Event()

        async def work() -> None:
            started.set()
            await asyncio.sleep(0.01)

        jobs = JobService(max_concurrent=1)
        jobs.schedule("tracked", work)

        assert jobs.active_count == 1
        await jobs.wait_for_all(timeout=5.0)
        assert jobs.active_count == 0

    async def test_wait_for_all_with_no_jobs_returns_immediately(self) -> None:
        await JobService().wait_for_all(timeout=0.1)
