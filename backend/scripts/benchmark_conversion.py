"""Measure the three conversion extraction paths against each other.

Compares a single call, fixed-size windows, and adaptively sized windows, over two
document shapes that behave completely differently: prose-like (a few records per
thousand tokens) and record-dense (a register, one per line). Running only one of them is
how a fixed 40,000-token window came to look correct while being eight times too large
for half the corpus.

Run this before changing ``CONVERSION_CHUNK_WINDOW_TOKENS`` or
``CONVERSION_CHUNK_MIN_WINDOW_TOKENS``. The sizing arithmetic is imported from
``app.ai.windows``, not reimplemented here, so what this measures is what the service
actually does.

**Do not assume windowing is faster.** It is not expected to be, for a document that
already fits: it makes more requests, pays the per-request overhead more times, and adds a
schema-inference call of its own. What it buys is completeness — a single call has to emit
every record it finds inside ``ai_max_output_tokens``, and past some document size it
simply cannot, so the surplus records are dropped. The number worth watching here is
therefore ``records``, not ``seconds``. A windowed run that is twice as slow and finds
three times as many records is the win this feature exists for; one that is slower and
finds the same records means the threshold is set too low.

Two modes:

* ``--fake`` (default) — a deterministic in-process provider. Measures call count, window
  layout, coverage and record totals with no network, no key and no cost. This is the
  mode to run in CI or while iterating.
* ``--live`` — the real Gemini API, for wall-clock timings. Costs money, needs
  ``GEMINI_API_KEY``, and is the only mode that can answer "is it faster".

Usage::

    python scripts/benchmark_conversion.py
    python scripts/benchmark_conversion.py --sizes 5000,40000,120000
    python scripts/benchmark_conversion.py --live --document <uuid> --owner <owner-id>
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import Iterable
from dataclasses import dataclass, replace
from uuid import UUID, uuid4

from app.ai.windows import (
    ContextWindow,
    adaptive_window_tokens,
    output_tokens_per_record,
    plan_windows,
)
from app.repositories.chunk_repo import Chunk

#: Roughly one record's worth of text, the shape the extractor sees in a real register.
_RECORD_TEMPLATE = (
    "ITEM {index}: reference=REF-{index:05d} condition=Good status=Active "
    "location=Bay-{index} note=Inspected and returned to service. "
)

#: Chunk size the ingestion chunker lands on for prose, near enough for planning.
_CHUNK_TOKENS = 400

#: The two document shapes this compares, in records per 1,000 source tokens.
#:
#: A contract or a report states a few facts per page; a register or a ledger states one
#: per line. They are the same document to the input context and eight times apart in what
#: the reply has to carry, which is the entire reason window size cannot be a constant.
PROSE_DENSITY = 3.0
DENSE_DENSITY = 25.0


@dataclass(frozen=True, slots=True)
class Measurement:
    """One row of the comparison table."""

    label: str
    document_tokens: int
    calls: int
    windows: int
    records: int
    coverage: float
    seconds: float

    def as_row(self) -> str:
        return (
            f"{self.label:<22}{self.document_tokens:>12,}{self.calls:>8}{self.windows:>9}"
            f"{self.records:>10,}{self.coverage:>11.0%}{self.seconds:>10.2f}"
        )


def _synthetic_chunks(document_tokens: int, *, density: float) -> list[Chunk]:
    """A document of a given size, as ingestion would have stored it.

    Args:
        document_tokens: Total estimated size.
        density: Records per 1,000 tokens of source. This is the variable that actually
            decides the right window size, and it is a property of the document rather
            than of the system: a register of transactions might hold 30 records per
            1,000 tokens, a contract two or three. A window sized for one is wrong for
            the other, which is why it is a knob here and a setting in production.
    """
    chunk_count = max(1, document_tokens // _CHUNK_TOKENS)
    records_per_chunk = max(1, round(_CHUNK_TOKENS * density / 1_000))

    chunks: list[Chunk] = []
    for order in range(chunk_count):
        first = order * records_per_chunk
        text = "".join(
            _RECORD_TEMPLATE.format(index=first + offset) for offset in range(records_per_chunk)
        )
        chunks.append(
            Chunk(
                id=uuid4(),
                order=order,
                text=text,
                block_ids=[uuid4()],
                token_estimate=_CHUNK_TOKENS,
                char_count=len(text),
                location_label=f"page {order + 1}",
            )
        )
    return chunks


def _records_a_reply_can_carry(max_output_tokens: int, field_count: int) -> int:
    """How many records fit in one structured reply.

    Uses the application's own per-record cost, so the benchmark and the runtime cannot
    drift apart — a second copy of this arithmetic living here is exactly how a benchmark
    ends up certifying a threshold the code never uses.
    """
    return max(1, int(max_output_tokens / output_tokens_per_record(field_count)))


def _extract(chunks: Iterable[Chunk], ceiling: int) -> int:
    """What one call returns: every record in the text it was sent, up to what fits.

    One rule, and it is the whole behaviour under test — it is what a single call does
    badly on a long document, and what windowing exists to work around.
    """
    return min(sum(chunk.text.count("ITEM ") for chunk in chunks), ceiling)


def _measure_fake(
    document_tokens: int,
    *,
    window_tokens: int,
    min_window_tokens: int,
    max_output_tokens: int,
    density: float,
    field_count: int,
) -> list[Measurement]:
    """Model all three paths without a network."""
    chunks = _synthetic_chunks(document_tokens, density=density)
    total_records = sum(chunk.text.count("ITEM ") for chunk in chunks)
    ceiling = _records_a_reply_can_carry(max_output_tokens, field_count)

    def measure(
        label: str, windows: list[ContextWindow], records: int, extra_calls: int
    ) -> Measurement:
        return Measurement(
            label=label,
            document_tokens=document_tokens,
            # Schema inference, one call per window, then the plan and the proposal.
            calls=1 + len(windows) + 2 + extra_calls,
            windows=len(windows),
            records=records,
            coverage=records / total_records if total_records else 1.0,
            seconds=0.0,
        )

    # --- single call ---------------------------------------------------------
    start = time.perf_counter()
    single = measure("single call", [ContextWindow(0, tuple(chunks))], _extract(chunks, ceiling), 0)
    single = replace(single, calls=4, seconds=time.perf_counter() - start)

    # --- fixed windows, the size before adaptive sizing existed ---------------
    start = time.perf_counter()
    fixed_windows = plan_windows(chunks, window_tokens=window_tokens)
    fixed = measure(
        f"windowed (fixed {window_tokens // 1000}k)",
        fixed_windows,
        sum(_extract(window.chunks, ceiling) for window in fixed_windows),
        0,
    )
    fixed = replace(fixed, seconds=time.perf_counter() - start)

    # --- adaptive -------------------------------------------------------------
    # Window 0 is planned at the floor, extracted, and its answer sizes the rest. Exactly
    # what `AnalysisService._extract_adaptively` does, against the same function.
    start = time.perf_counter()
    first = plan_windows(chunks, window_tokens=min_window_tokens)[0]
    observed = _extract(first.chunks, ceiling)
    sized = adaptive_window_tokens(
        observed_records=observed,
        observed_tokens=first.token_estimate,
        field_count=field_count,
        max_output_tokens=max_output_tokens,
        min_window_tokens=min_window_tokens,
        max_window_tokens=window_tokens,
    )
    tail = plan_windows(chunks[len(first.chunks) :], window_tokens=sized, start_index=1)
    adaptive = measure(
        f"adaptive (-> {sized:,})",
        [first, *tail],
        observed + sum(_extract(window.chunks, ceiling) for window in tail),
        0,
    )
    adaptive = replace(adaptive, seconds=time.perf_counter() - start)

    # --- what the service actually does ---------------------------------------
    # Above the threshold it windows on size. Below it, it makes the single call and
    # falls back to windowing only if that reply ran out of room — so the row below is
    # the single call's own result when it fit, and the adaptive result plus one wasted
    # call when it did not. This is the row to read when asking "what will happen".
    over_threshold = document_tokens > window_tokens
    overran = single.records < total_records
    if over_threshold:
        service = replace(adaptive, label="service (windowed)")
    elif overran:
        service = replace(
            adaptive,
            label="service (fell back)",
            # One doomed call before the windowed run begins.
            calls=adaptive.calls + 1,
        )
    else:
        service = replace(single, label="service (single call)")

    return [single, fixed, adaptive, service]


async def _measure_live(document_id: UUID, owner: str, *, window_tokens: int) -> None:
    """Time both paths against the real API, on a real stored document.

    Deliberately separate from the fake mode and never mixed into it: this one spends
    money and its numbers vary run to run.
    """
    # Imported here rather than at module scope so the fake mode needs no database, no
    # settings file and no provider — it must stay runnable anywhere.
    from app.api.deps import _build_provider, get_settings
    from app.core.database import get_session_factory
    from app.repositories.ai_result_repo import AIResultRepository
    from app.repositories.block_repo import ContentBlockRepository
    from app.repositories.chunk_repo import ChunkRepository
    from app.repositories.document_repo import DocumentRepository
    from app.services.analysis_service import AnalysisService

    settings = get_settings()
    if not settings.ai_configured:
        raise SystemExit("Live mode needs a configured provider (set GEMINI_API_KEY).")

    factory = get_session_factory()
    instruction = "List every record in this document with all of its fields."

    print(f"{'path':<22}{'seconds':>10}{'records':>10}{'in_tokens':>12}{'out_tokens':>12}")
    for label, windowed in (("single call", False), ("windowed", True)):
        async with factory() as session:
            provider = _build_provider(settings)
            analysis = AnalysisService(
                provider=provider,
                settings=settings,
                documents=DocumentRepository(session),
                chunks=ChunkRepository(session),
                blocks=ContentBlockRepository(session),
                results=AIResultRepository(session),
            )
            start = time.perf_counter()
            if windowed:
                result = await analysis.structure_windowed(
                    document_id,
                    owner_id=owner,
                    instruction=instruction,
                    window_tokens=window_tokens,
                    concurrency=settings.conversion_chunk_concurrency,
                    min_window_tokens=settings.conversion_chunk_min_window_tokens,
                    refresh=True,
                )
            else:
                result = await analysis.structure(
                    document_id, owner_id=owner, instruction=instruction, refresh=True
                )
            elapsed = time.perf_counter() - start
            insight = result.insight
            print(
                f"{label:<22}{elapsed:>10.2f}{len(insight.records):>10,}"
                f"{insight.input_tokens:>12,}{insight.output_tokens:>12,}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        default="10000,40000,80000,200000,500000",
        help="Comma-separated document sizes in estimated tokens.",
    )
    parser.add_argument("--window-tokens", type=int, default=40_000)
    parser.add_argument("--min-window-tokens", type=int, default=2_000)
    parser.add_argument("--max-output-tokens", type=int, default=8_000)
    parser.add_argument("--fields", type=int, default=5, help="Fields in the schema.")
    parser.add_argument(
        "--density",
        type=float,
        default=None,
        help=(
            "Records per 1,000 source tokens. High for a register or ledger, low for "
            "prose. Omit to run both a prose-like (3) and a record-dense (25) document."
        ),
    )
    parser.add_argument("--live", action="store_true", help="Hit the real API (costs money).")
    parser.add_argument("--document", help="Document UUID, required with --live.")
    parser.add_argument("--owner", default="local-owner", help="Owner id, with --live.")
    args = parser.parse_args()

    if args.live:
        if not args.document:
            raise SystemExit("--live requires --document <uuid>")
        asyncio.run(
            _measure_live(UUID(args.document), args.owner, window_tokens=args.window_tokens)
        )
        return

    fits = _records_a_reply_can_carry(args.max_output_tokens, args.fields)
    # Both by default: one number for one density is what produced a threshold that was
    # right for prose and eight times too large for a register.
    densities = [args.density] if args.density is not None else [PROSE_DENSITY, DENSE_DENSITY]

    print(
        f"max_window={args.window_tokens:,}   min_window={args.min_window_tokens:,}   "
        f"max_output={args.max_output_tokens:,}   fields={args.fields}\n"
        f"one record costs ~{output_tokens_per_record(args.fields)} output tokens, "
        f"so one reply holds ~{fits} records\n"
    )
    header = (
        f"{'path':<22}{'doc tokens':>12}{'calls':>8}{'windows':>9}"
        f"{'records':>10}{'coverage':>11}{'seconds':>10}"
    )

    for density in densities:
        kind = "prose-like" if density <= 10 else "record-dense"
        print(f"=== {kind}: {density:g} records per 1,000 tokens ===\n")
        print(header)
        print("-" * len(header))

        for raw in args.sizes.split(","):
            size = int(raw.strip())
            for row in _measure_fake(
                size,
                window_tokens=args.window_tokens,
                min_window_tokens=args.min_window_tokens,
                max_output_tokens=args.max_output_tokens,
                density=density,
                field_count=args.fields,
            ):
                print(row.as_row())
            print()

    print(
        "coverage = records returned / records present. Read the three rows together:\n"
        "\n"
        "  single call  falls off as soon as the document holds more records than one\n"
        "               reply can carry. This is the failure the whole feature is for.\n"
        "  fixed        divides the document but not by enough at high density: each\n"
        "               window is now hitting the same ceiling on its own. One constant\n"
        "               cannot serve both shapes, which is what the two tables show.\n"
        "  adaptive     sizes windows from density measured in the first window and the\n"
        "               schema's field count, and should read 100% in both tables.\n"
        "\n"
        "The cost of that is in the calls column, and it is not small at high density.\n"
        "That is the trade this feature makes: more requests, all of the records.\n"
        "\n"
        "  service      what the running code does. Above the threshold it windows on\n"
        "               size. Below it, it makes the single call and falls back to\n"
        "               windowing only when that reply ran out of room — the extra call\n"
        "               in the count is the doomed one. It should read 100% throughout."
    )


if __name__ == "__main__":
    main()
