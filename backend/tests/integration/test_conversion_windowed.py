"""Tests for windowed conversion extraction — the large-document path.

**No test here calls a real API** (CLAUDE.md §7). Real repositories, a real database
session and a real `AnalysisService`; only the model is replaced.

The fake below is deliberately not a canned-reply machine. It reads the context blocks it
was handed and answers *from them*, the way a real extraction does. That is what lets
these tests prove the properties that matter and could not otherwise be distinguished
from luck:

* a window that was never sent a record cannot return it, so **nothing is invented across
  windows** — the fake is physically unable to do it;
* the merged order must match document order, because each record's identity is baked
  into the chunk it came from;
* a quote only verifies if it really appears in stored content, so **provenance surviving
  the merge** is a fact about the data rather than about the fake.

The document is built as one chunk per item, sized so that the window budget divides it
into several windows.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.conversion_schema import PlannedDocument, ProposedCorrectionList
from app.ai.dynamic_schema import InferredStructureField, InferredStructureSchema
from app.ai.windows import output_tokens_per_record
from app.core.config import Settings
from app.core.exceptions import (
    AIUnavailableError,
    InsufficientContentError,
    InvalidModelResponseError,
    OutputTruncatedError,
)
from app.models.domain.content import BlockType, ContentBlock, Provenance, ProvenanceKind
from app.models.domain.document import Document, ProcessingStatus, SupportedFileType
from app.models.domain.insight import ContextBlock, ModelResult
from app.repositories.ai_result_repo import AIResultRepository
from app.repositories.block_repo import ContentBlockRepository
from app.repositories.chunk_repo import ChunkRepository
from app.repositories.document_repo import DocumentRepository
from app.services.analysis_service import AnalysisService
from app.services.conversion_service import ConversionService
from tests.fixtures.fake_provider import FakeLLMProvider, RecordedCall

OWNER = "owner-a"

#: Items in the synthetic document. Enough to need several windows at the test budget.
ITEM_COUNT = 12

#: Item 7 states no condition. It must still state none after merging — a value the
#: source omits cannot be supplied by a window that happened to see a different item.
ITEM_WITHOUT_CONDITION = 7

INSTRUCTION = "List every item with its reference and condition, grouped by condition."


def _item_text(index: int) -> str:
    """One item's paragraph. Padded so each chunk is worth roughly 60 tokens."""
    condition = "" if index == ITEM_WITHOUT_CONDITION else f" condition=Good-{index}"
    body = f"ITEM {index}: reference=REF-{index:03d}{condition}."
    return body + " " + ("filler " * 40)


@dataclass
class _WindowAwareProvider(FakeLLMProvider):
    """Answers each call from the context it was actually given.

    Dispatches on ``output_model`` rather than on call order, because window extractions
    run concurrently and a positional sequence would be a race rather than a fixture.

    Attributes:
        fail_items: Items whose window should raise instead of answering, simulating a
            provider failure confined to part of a document.
        max_in_flight: High-water mark of concurrent extraction calls, for proving the
            semaphore actually limits anything.
    """

    fail_items: frozenset[int] = frozenset()
    max_in_flight: int = 0
    _in_flight: int = 0
    extraction_calls: list[tuple[int, ...]] = field(default_factory=list)
    #: Records returned per item present in the window. Raising it makes the document
    #: denser without changing a byte of its text, which is exactly the variable adaptive
    #: sizing responds to.
    records_per_item: int = 1
    #: Tokens each extraction call was sent, in call order — how a test sees the window
    #: size the service actually chose.
    window_token_sizes: list[int] = field(default_factory=list)
    #: Records one reply can carry before it runs out of room. Effectively unlimited by
    #: default, so only the tests that care about the ceiling meet it.
    reply_record_limit: int = 10**9

    async def generate_structured(
        self,
        *,
        system: str,
        instruction: str,
        context_blocks: list[ContextBlock],
        document_title: str,
        output_model: type[BaseModel],
    ) -> ModelResult:
        self.calls.append(
            RecordedCall(
                system=system,
                instruction=instruction,
                context_blocks=list(context_blocks),
                document_title=document_title,
                enable_citations=False,
                output_model=output_model,
            )
        )

        if output_model is InferredStructureSchema:
            return _result(_inferred_schema())
        if output_model is PlannedDocument:
            return _result(_plan())
        if output_model is ProposedCorrectionList:
            return _result(ProposedCorrectionList(corrections=[]))

        return await self._extract(context_blocks)

    async def _extract(self, context_blocks: list[ContextBlock]) -> ModelResult:
        """Return one record per item present in *this window's* text."""
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            # A real network call yields; without a yield here every coroutine would run
            # to completion before the next started and the semaphore would never be
            # observed doing anything.
            await asyncio.sleep(0)

            indices = tuple(
                index
                for index in range(ITEM_COUNT)
                if any(f"ITEM {index}:" in block.text for block in context_blocks)
            )
            self.extraction_calls.append(indices)
            self.window_token_sizes.append(
                sum(max(1, len(block.text) // 4) for block in context_blocks)
            )

            if self.error is not None:
                raise self.error

            # Scoped to calls that cover part of the document, so a test can inject a
            # failure into one *window* without also breaking the whole-document call
            # that precedes it on the fallback path.
            covers_whole_document = len(indices) == ITEM_COUNT
            if not covers_whole_document and any(index in self.fail_items for index in indices):
                raise AIUnavailableError("The AI service is busy right now.")

            # Stands in for the provider's MAX_TOKENS check: a call asked for more
            # records than one reply can carry, so it produces nothing usable rather
            # than a convincing fraction.
            if len(indices) * self.records_per_item > self.reply_record_limit:
                raise OutputTruncatedError("The AI ran out of room before it finished answering.")

            return _result(
                SimpleNamespace(
                    records=[
                        _record_for(index)
                        for index in indices
                        for _ in range(self.records_per_item)
                    ],
                )
            )
        finally:
            self._in_flight -= 1


def _result(parsed: Any) -> ModelResult:
    return ModelResult(parsed=parsed, input_tokens=100, output_tokens=50, stop_reason="end_turn")


def _value(value: Any, quote: str) -> SimpleNamespace:
    return SimpleNamespace(value=value, source_quote=quote)


def _record_for(index: int) -> SimpleNamespace:
    """One extracted record, quoting text that genuinely appears in item ``index``."""
    if index == ITEM_WITHOUT_CONDITION:
        # The source states no condition, so the extraction states none. Quoting nothing
        # is how "absent" is expressed — inventing a plausible value is the failure this
        # whole feature is written to avoid.
        return SimpleNamespace(
            reference=_value(f"REF-{index:03d}", f"reference=REF-{index:03d}"),
            condition=_value("", ""),
        )
    return SimpleNamespace(
        reference=_value(f"REF-{index:03d}", f"reference=REF-{index:03d}"),
        condition=_value(f"Good-{index}", f"condition=Good-{index}"),
    )


def _inferred_schema() -> InferredStructureSchema:
    return InferredStructureSchema(
        fields=[
            InferredStructureField(name="reference", type="string"),
            InferredStructureField(name="condition", type="string"),
        ]
    )


def _plan() -> PlannedDocument:
    return PlannedDocument.model_validate(
        {
            "title": "Item Register",
            "sections": [
                {
                    "heading": "All items",
                    "kind": "table",
                    "columns": ["reference", "condition"],
                }
            ],
        }
    )


@pytest.fixture
async def document(db_session: AsyncSession) -> Document:
    """A document of `ITEM_COUNT` items, one content block and one chunk each."""
    documents = DocumentRepository(db_session)
    created = await documents.create(
        owner_id=OWNER,
        original_filename="register.txt",
        file_type=SupportedFileType.TXT,
        size_bytes=sum(len(_item_text(index)) for index in range(ITEM_COUNT)),
        status=ProcessingStatus.READY,
    )

    blocks = ContentBlockRepository(db_session)
    block_ids = await blocks.replace_for_document(
        created.id,
        [
            ContentBlock(
                block_type=BlockType.PARAGRAPH,
                text=_item_text(index),
                order=index,
                provenance=Provenance(
                    kind=ProvenanceKind.PAGE, label=f"page {index + 1}", page=index + 1
                ),
            )
            for index in range(ITEM_COUNT)
        ],
    )

    # One chunk per item, so a window boundary always falls between whole items and the
    # window a record came from is unambiguous.
    await ChunkRepository(db_session).replace_for_document(
        created.id,
        [
            (_item_text(index), [block_ids[index]], 60, f"page {index + 1}")
            for index in range(ITEM_COUNT)
        ],
    )
    await db_session.flush()
    return created


@pytest.fixture
def build_service(db_session: AsyncSession, settings: Settings):  # type: ignore[no-untyped-def]
    def _build(
        provider: FakeLLMProvider,
        *,
        window_tokens: int = 200,
        concurrency: int = 3,
        enabled: bool = True,
        min_window_tokens: int = 200,
        max_output_tokens: int = 8_000,
    ) -> ConversionService:
        tuned = settings.model_copy(
            update={
                "conversion_chunk_window_tokens": window_tokens,
                "conversion_chunk_min_window_tokens": min_window_tokens,
                "conversion_chunk_concurrency": concurrency,
                "conversion_chunking_enabled": enabled,
                "ai_max_output_tokens": max_output_tokens,
            }
        )
        analysis = AnalysisService(
            provider=provider,
            settings=tuned,
            documents=DocumentRepository(db_session),
            chunks=ChunkRepository(db_session),
            blocks=ContentBlockRepository(db_session),
            results=AIResultRepository(db_session),
        )
        return ConversionService(analysis=analysis, provider=provider, settings=tuned)

    return _build


def _extraction_call_count(provider: _WindowAwareProvider) -> int:
    return len(provider.extraction_calls)


class TestPathSelection:
    """Which route a document takes, and that the choice is made on content size."""

    async def test_a_small_document_uses_the_single_call_path(
        self, build_service, document: Document
    ) -> None:
        """Below the threshold nothing changes: one extraction call, as before."""
        provider = _WindowAwareProvider()
        # 12 chunks x 60 tokens = 720, comfortably under this budget.
        service = build_service(provider, window_tokens=40_000)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        assert _extraction_call_count(provider) == 1
        assert provider.extraction_calls[0] == tuple(range(ITEM_COUNT))
        assert result.source_record_count == ITEM_COUNT

    async def test_a_large_document_uses_windowed_extraction(
        self, build_service, document: Document
    ) -> None:
        provider = _WindowAwareProvider()
        service = build_service(provider, window_tokens=200)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        # 60 tokens per chunk, 200-token budget -> 3 chunks per window, 4 windows.
        assert _extraction_call_count(provider) == 4
        assert result.source_record_count == ITEM_COUNT

    async def test_the_kill_switch_forces_the_single_call_path(
        self, build_service, document: Document
    ) -> None:
        """`conversion_chunking_enabled=false` restores the pre-windowing behaviour
        exactly, without needing the threshold to be retuned."""
        provider = _WindowAwareProvider()
        service = build_service(provider, window_tokens=200, enabled=False)

        await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        assert _extraction_call_count(provider) == 1

    async def test_a_document_that_fits_one_window_is_not_windowed(
        self, build_service, document: Document
    ) -> None:
        """The threshold is the window size, so "fits in one window" and "takes the old
        path" are the same condition rather than two that could drift apart."""
        provider = _WindowAwareProvider()
        service = build_service(provider, window_tokens=720)

        await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        assert _extraction_call_count(provider) == 1


class TestWindowBoundaries:
    """What each window is actually sent."""

    async def test_no_window_receives_content_from_another(
        self, build_service, document: Document
    ) -> None:
        provider = _WindowAwareProvider()
        service = build_service(provider, window_tokens=200)

        await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        seen = [index for call in provider.extraction_calls for index in call]
        assert sorted(seen) == list(range(ITEM_COUNT))
        assert len(seen) == len(set(seen)), "an item was sent to more than one window"

    async def test_windows_cover_the_document_in_order(
        self, build_service, document: Document
    ) -> None:
        provider = _WindowAwareProvider()
        service = build_service(provider, window_tokens=200)

        await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        # Sorted because the calls complete concurrently and may be recorded in any
        # order; what must hold is that each window is a contiguous run of items.
        for call in provider.extraction_calls:
            assert list(call) == list(range(call[0], call[0] + len(call)))

    async def test_a_chunk_is_never_split_across_windows(
        self, build_service, document: Document
    ) -> None:
        """Each chunk holds one whole item. If a chunk were split, an item's text would
        arrive in two windows and be extracted twice."""
        provider = _WindowAwareProvider()
        service = build_service(provider, window_tokens=200)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        assert result.source_record_count == ITEM_COUNT


class TestMerging:
    """What the merged record list looks like once the windows come back."""

    async def test_records_are_in_document_order_after_merging(
        self, build_service, document: Document
    ) -> None:
        provider = _WindowAwareProvider()
        service = build_service(provider, window_tokens=200)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        section = result.document.sections[0]
        references = [
            next(f.value.text for f in record.fields if f.name == "reference")
            for record in section.records
        ]
        assert references == [f"REF-{index:03d}" for index in range(ITEM_COUNT)]

    async def test_record_indices_are_positional_and_assigned_after_the_merge(
        self, build_service, document: Document
    ) -> None:
        """Nothing downstream can tell how many calls produced the list."""
        provider = _WindowAwareProvider()
        service = build_service(provider, window_tokens=200)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        indices = [record.source_record_index for record in result.document.sections[0].records]
        assert indices == list(range(ITEM_COUNT))

    async def test_provenance_survives_the_merge(self, build_service, document: Document) -> None:
        """Each record must still point at its own page, not at the window's first page.

        This is the assertion that would fail if provenance were rebuilt from a position
        inside the window rather than carried through as the absolute location it is.
        """
        provider = _WindowAwareProvider()
        service = build_service(provider, window_tokens=200)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        for index, record in enumerate(result.document.sections[0].records):
            reference = next(f for f in record.fields if f.name == "reference")
            assert reference.value.citation is not None
            assert reference.value.citation.provenance is not None
            assert reference.value.citation.provenance.label == f"page {index + 1}"

    async def test_a_value_the_source_omits_stays_omitted(
        self, build_service, document: Document
    ) -> None:
        """Requirement 8, stated as a test: a field missing from one window must not be
        filled in from an unrelated one."""
        provider = _WindowAwareProvider()
        service = build_service(provider, window_tokens=200)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        record = result.document.sections[0].records[ITEM_WITHOUT_CONDITION]
        condition = next(f for f in record.fields if f.name == "condition")
        assert condition.value.text == ""
        # And specifically not borrowed from a neighbour in another window.
        assert "Good-" not in condition.value.text

    async def test_the_result_says_the_document_was_read_in_parts(
        self, build_service, document: Document
    ) -> None:
        provider = _WindowAwareProvider()
        service = build_service(provider, window_tokens=200)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        assert any("read in 4 parts" in warning for warning in result.warnings)


class TestPartialFailure:
    """One window failing must not discard the windows that worked, or hide that it did."""

    async def test_records_from_successful_windows_survive(
        self, build_service, document: Document
    ) -> None:
        provider = _WindowAwareProvider(fail_items=frozenset({0}))
        service = build_service(provider, window_tokens=200)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        # The first window (items 0-2) failed; the remaining nine items survive.
        assert result.source_record_count == ITEM_COUNT - 3

    async def test_a_partial_result_says_so(self, build_service, document: Document) -> None:
        """A partial result that looks whole is worse than a failure: nothing about the
        records themselves reveals that part of the document is missing from them."""
        provider = _WindowAwareProvider(fail_items=frozenset({0}))
        service = build_service(provider, window_tokens=200)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        assert any(
            "1 of 4 parts of this document could not be read" in warning
            for warning in result.warnings
        )

    async def test_surviving_records_keep_their_own_provenance(
        self, build_service, document: Document
    ) -> None:
        """A gap must not shift the records after it onto the wrong pages."""
        provider = _WindowAwareProvider(fail_items=frozenset({0}))
        service = build_service(provider, window_tokens=200)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        first = result.document.sections[0].records[0]
        reference = next(f for f in first.fields if f.name == "reference")
        # Item 3 is the first survivor, and it must still report page 4.
        assert reference.value.text == "REF-003"
        assert reference.value.citation is not None
        assert reference.value.citation.provenance is not None
        assert reference.value.citation.provenance.label == "page 4"

    async def test_every_window_failing_is_reported_as_a_failure(
        self, build_service, document: Document
    ) -> None:
        """Not as an empty success, which is what a bare merge of zero results is."""
        provider = _WindowAwareProvider(fail_items=frozenset(range(ITEM_COUNT)))
        service = build_service(provider, window_tokens=200)

        with pytest.raises(AIUnavailableError):
            await service.convert(document.id, INSTRUCTION, owner_id=OWNER)


class TestRateLimits:
    """Concurrency is bounded, and no second retry system was added."""

    async def test_concurrency_never_exceeds_the_configured_limit(
        self, build_service, document: Document
    ) -> None:
        provider = _WindowAwareProvider()
        service = build_service(provider, window_tokens=60, concurrency=2)

        await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        assert _extraction_call_count(provider) == ITEM_COUNT
        # `== 2`, not `<= 2`: the limit must actually be reached, or this test would
        # also pass against an implementation that never ran anything in parallel.
        assert provider.max_in_flight == 2

    async def test_a_concurrency_of_one_serialises_the_windows(
        self, build_service, document: Document
    ) -> None:
        """The setting a rate-limited deployment would fall back to."""
        provider = _WindowAwareProvider()
        service = build_service(provider, window_tokens=60, concurrency=1)

        await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        assert provider.max_in_flight == 1

    async def test_a_failing_window_is_not_retried_by_this_layer(
        self, build_service, document: Document
    ) -> None:
        """Retries belong to the provider, which already has a budget for 429s. A second
        one here would multiply into `concurrency x retries` requests inside the same
        rate-limit window."""
        provider = _WindowAwareProvider(fail_items=frozenset({0}))
        service = build_service(provider, window_tokens=200, concurrency=1)

        await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        # Four windows, four extraction attempts. The failed one is attempted once.
        assert _extraction_call_count(provider) == 4


class TestCachingAndSafety:
    """The merged result goes through everything a single-call result does."""

    async def test_a_repeat_conversion_reuses_the_windowed_extraction(
        self, build_service, document: Document
    ) -> None:
        provider = _WindowAwareProvider()
        service = build_service(provider, window_tokens=200)

        await service.convert(document.id, INSTRUCTION, owner_id=OWNER)
        first_round = _extraction_call_count(provider)
        second = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        assert _extraction_call_count(provider) == first_round
        assert second.records_from_cache is True

    async def test_refresh_re_reads_every_window(self, build_service, document: Document) -> None:
        provider = _WindowAwareProvider()
        service = build_service(provider, window_tokens=200)

        await service.convert(document.id, INSTRUCTION, owner_id=OWNER)
        await service.convert(document.id, INSTRUCTION, owner_id=OWNER, refresh=True)

        assert _extraction_call_count(provider) == 8

    async def test_the_correction_validator_still_runs_once_over_everything(
        self, build_service, document: Document
    ) -> None:
        """The reason windowing stops at extraction. One review, over the whole merged
        list, with globally correct indices — rather than one per window whose indices
        would need remapping."""
        provider = _WindowAwareProvider()
        service = build_service(provider, window_tokens=200)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        proposal_calls = [
            call for call in provider.calls if call.output_model is ProposedCorrectionList
        ]
        assert len(proposal_calls) == 1
        assert result.review is not None

    async def test_the_plan_is_generated_once_not_per_window(
        self, build_service, document: Document
    ) -> None:
        provider = _WindowAwareProvider()
        service = build_service(provider, window_tokens=200)

        await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        plan_calls = [call for call in provider.calls if call.output_model is PlannedDocument]
        assert len(plan_calls) == 1


class TestAdaptiveSizing:
    """Window size responds to how densely the document packs records.

    The two documents below are the *same text*: only how many records the extraction
    finds in it differs, which is what makes this a test of the sizing decision rather
    than of the fixture. `window_tokens` is the ceiling here (600, so 12 chunks of 60
    tokens must be windowed) and `min_window_tokens` the floor (120, two chunks).
    """

    async def test_a_prose_like_document_widens_its_windows(
        self, build_service, document: Document
    ) -> None:
        """One record per item. The opening window finds few, so the rest of the
        document is read in windows as large as the ceiling allows."""
        provider = _WindowAwareProvider(records_per_item=1)
        service = build_service(provider, window_tokens=600, min_window_tokens=120)

        await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        # Window 0 is the floor (2 chunks); the remaining 10 fit one 600-token window.
        assert provider.extraction_calls[0] == (0, 1)
        assert len(provider.extraction_calls) == 2

    async def test_a_record_dense_document_keeps_its_windows_small(
        self, build_service, document: Document
    ) -> None:
        """Forty records per item over the same text. The opening window measures that
        density, and every window after it stays at the floor."""
        provider = _WindowAwareProvider(records_per_item=40)
        service = build_service(provider, window_tokens=600, min_window_tokens=120)

        await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        assert len(provider.extraction_calls) == 6  # 12 chunks / 2 per window
        for call in provider.extraction_calls:
            assert len(call) == 2

    async def test_density_alone_decides_the_difference(
        self, build_service, document: Document
    ) -> None:
        """Same document, same settings, same schema — only record density differs, and
        the dense run reads the document in more, smaller pieces."""
        sparse = _WindowAwareProvider(records_per_item=1)
        await build_service(sparse, window_tokens=600, min_window_tokens=120).convert(
            document.id, INSTRUCTION, owner_id=OWNER
        )

        dense = _WindowAwareProvider(records_per_item=40)
        await build_service(dense, window_tokens=600, min_window_tokens=120).convert(
            document.id, INSTRUCTION, owner_id=OWNER, refresh=True
        )

        assert len(dense.extraction_calls) > len(sparse.extraction_calls)
        assert max(dense.window_token_sizes) <= max(sparse.window_token_sizes)

    async def test_no_window_is_planned_to_overrun_the_output_ceiling(
        self, build_service, document: Document
    ) -> None:
        """The property the whole feature exists for, asserted against the real pipeline.

        Every window after the first is sized from a measured density, so the records it
        is expected to return must cost less than the ceiling. Checked with the same
        per-record cost the service used.
        """
        provider = _WindowAwareProvider(records_per_item=40)
        service = build_service(
            provider, window_tokens=600, min_window_tokens=120, max_output_tokens=8_000
        )

        await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        per_record = output_tokens_per_record(2)  # reference + condition
        for call, sent in zip(provider.extraction_calls, provider.window_token_sizes, strict=True):
            returned = len(call) * 40
            assert returned * per_record <= 8_000, (
                f"a window of {sent} tokens returned {returned} records"
            )

    async def test_a_document_with_no_records_in_its_opening_still_completes(
        self, build_service, document: Document
    ) -> None:
        """No density measurement is possible, so the maximum is used — the size this
        document would have had before adaptive sizing existed. It must not divide by
        zero, stall, or drop the rest of the document."""
        provider = _WindowAwareProvider(records_per_item=0)
        service = build_service(provider, window_tokens=600, min_window_tokens=120)

        # Every window returns nothing, so the extraction legitimately finds no records.
        with pytest.raises(InsufficientContentError):
            await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        # The whole document was still read rather than abandoned after the first window.
        seen = [index for call in provider.extraction_calls for index in call]
        assert sorted(seen) == list(range(ITEM_COUNT))

    async def test_ordering_and_provenance_survive_adaptive_windows(
        self, build_service, document: Document
    ) -> None:
        """Re-planning the tail at a different size must not disturb either. Window 0 is
        two chunks and the rest are ten — an unequal split, which is precisely the shape
        that would expose an index or offset mistake."""
        provider = _WindowAwareProvider(records_per_item=1)
        service = build_service(provider, window_tokens=600, min_window_tokens=120)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        records = result.document.sections[0].records
        assert [record.source_record_index for record in records] == list(range(ITEM_COUNT))
        for index, record in enumerate(records):
            reference = next(f for f in record.fields if f.name == "reference")
            assert reference.value.text == f"REF-{index:03d}"
            assert reference.value.citation is not None
            assert reference.value.citation.provenance is not None
            assert reference.value.citation.provenance.label == f"page {index + 1}"


class TestOutputCeilingFallback:
    """A document short enough for one call whose *answer* is too big for one reply.

    Length and output size are different quantities, and the second is the one that
    matters. Every document here is 720 tokens against a 40,000-token threshold — far
    below it, and never windowed on size — so what routes it is entirely what came back.

    `reply_record_limit` stands in for the provider's MAX_TOKENS check, which is what a
    real overrun produces since the `finishReason` fix.
    """

    async def test_a_below_threshold_document_that_answers_cleanly_is_untouched(
        self, build_service, document: Document
    ) -> None:
        """The path that must not change. One extraction call, no windowing, no probe."""
        provider = _WindowAwareProvider(records_per_item=1, reply_record_limit=1_000)
        service = build_service(provider, window_tokens=40_000, min_window_tokens=120)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        assert len(provider.extraction_calls) == 1
        assert provider.extraction_calls[0] == tuple(range(ITEM_COUNT))
        assert result.source_record_count == ITEM_COUNT

    async def test_an_overrunning_single_call_falls_back_to_windows(
        self, build_service, document: Document
    ) -> None:
        """40 records per item over 12 items is 480 records, which one reply cannot hold.
        Before this fallback the document was simply short enough to escape windowing."""
        provider = _WindowAwareProvider(records_per_item=40, reply_record_limit=100)
        service = build_service(provider, window_tokens=40_000, min_window_tokens=120)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        # One doomed call, then the windowed path reads the document in pieces.
        assert len(provider.extraction_calls) > 1
        assert provider.extraction_calls[0] == tuple(range(ITEM_COUNT))
        assert result.source_record_count == ITEM_COUNT * 40

    async def test_the_fallback_recovers_every_record(
        self, build_service, document: Document
    ) -> None:
        """The point of the exercise: the records the single call could not return."""
        provider = _WindowAwareProvider(records_per_item=40, reply_record_limit=100)
        service = build_service(provider, window_tokens=40_000, min_window_tokens=120)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        references = {
            next(f.value.text for f in record.fields if f.name == "reference")
            for record in result.document.sections[0].records
        }
        assert references == {f"REF-{index:03d}" for index in range(ITEM_COUNT)}

    async def test_ordering_and_provenance_survive_the_fallback(
        self, build_service, document: Document
    ) -> None:
        """The fallback re-runs extraction through a different route, so the guarantees
        have to be re-established on that route rather than assumed from the first."""
        # 12 records in one call overruns; the 10 in the windowed tail do not.
        provider = _WindowAwareProvider(records_per_item=1, reply_record_limit=11)
        service = build_service(provider, window_tokens=40_000, min_window_tokens=120)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        records = result.document.sections[0].records
        assert [record.source_record_index for record in records] == list(range(ITEM_COUNT))
        for index, record in enumerate(records):
            reference = next(f for f in record.fields if f.name == "reference")
            assert reference.value.text == f"REF-{index:03d}"
            assert reference.value.citation is not None
            assert reference.value.citation.provenance is not None
            assert reference.value.citation.provenance.label == f"page {index + 1}"

    async def test_a_schema_mismatch_does_not_trigger_the_fallback(
        self, build_service, document: Document
    ) -> None:
        """The reason the error is a distinct subclass. A reply that did not match the
        schema will not match it in smaller pieces either — re-running the whole document
        through windows would cost a great deal to reach the same failure."""
        provider = _WindowAwareProvider(records_per_item=1)
        provider.error = InvalidModelResponseError("The AI returned an unexpected shape.")
        service = build_service(provider, window_tokens=40_000, min_window_tokens=120)

        with pytest.raises(InvalidModelResponseError):
            await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

    async def test_a_document_already_at_the_floor_reports_rather_than_looping(
        self, build_service, document: Document
    ) -> None:
        """The terminal case. When the whole document is one minimum-sized window there
        is nothing left to divide, so the truncation propagates instead of the fallback
        calling itself forever."""
        provider = _WindowAwareProvider(records_per_item=40, reply_record_limit=10)
        service = build_service(provider, window_tokens=40_000, min_window_tokens=40_000)

        with pytest.raises(OutputTruncatedError):
            await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        # Two attempts at most: the original, and the one the fallback delegates back.
        assert len(provider.extraction_calls) <= 2

    async def test_the_fallback_uses_the_same_sizing_as_the_large_document_path(
        self, build_service, document: Document
    ) -> None:
        """No second formula. After falling back, window 0 is the floor and the rest are
        sized from its density — the same two-phase shape `structure_windowed` always
        uses, which is visible in the first window being smaller than those after it."""
        provider = _WindowAwareProvider(records_per_item=1, reply_record_limit=11)
        service = build_service(provider, window_tokens=40_000, min_window_tokens=120)

        await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        windowed = provider.extraction_calls[1:]  # skip the doomed single call
        assert len(windowed[0]) == 2  # the floor: 120 tokens is two 60-token chunks
        assert len(windowed[1]) > len(windowed[0])  # the tail, sized from what it found

    async def test_a_partial_failure_after_falling_back_is_still_reported(
        self, build_service, document: Document
    ) -> None:
        """Partial-failure behaviour is a property of the windowed path, so it has to
        hold when that path is reached this way too."""
        provider = _WindowAwareProvider(
            records_per_item=1, reply_record_limit=11, fail_items=frozenset({11})
        )
        service = build_service(provider, window_tokens=40_000, min_window_tokens=120)

        result = await service.convert(document.id, INSTRUCTION, owner_id=OWNER)

        assert any("could not be read" in warning for warning in result.warnings)
        assert result.source_record_count < ITEM_COUNT
