"""Regenerates the committed synthetic PDF fixtures.

Run from ``backend/``::

    python tests/fixtures/make_pdf_fixtures.py

The PDFs used by the test suite are committed as files because PDFs cannot be produced by
any library the project depends on, and adding a PDF *writer* (such as reportlab) purely
to build test inputs was judged not worth the dependency. This script writes them by
emitting the PDF format directly, so the fixtures stay reproducible and auditable rather
than being opaque binaries of unknown origin.

They are deliberately tiny and entirely synthetic — no real document is ever committed.
"""

from __future__ import annotations

from pathlib import Path

FIXTURES_DIR = Path(__file__).parent

#: Standard 14 font, so no font file needs embedding.
_FONT = "Helvetica"


def _escape(text: str) -> str:
    """Escape the three characters that are special inside a PDF string literal."""
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(pages: list[list[tuple[str, int, int, int]]]) -> bytes:
    """Build a minimal but valid PDF.

    Args:
        pages: One list per page of ``(text, x, y, font_size)`` items.

    Returns:
        The PDF file's bytes.
    """
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        """Append an object and return its 1-based number."""
        objects.append(body)
        return len(objects)

    # Object numbers are allocated up front because the page tree and its pages
    # reference each other.
    font_number = 1
    catalog_number = 2
    pages_number = 3
    objects.extend([b"", b"", b""])  # placeholders, filled in below

    objects[font_number - 1] = (
        b"<< /Type /Font /Subtype /Type1 /BaseFont /" + _FONT.encode() + b" >>"
    )

    page_numbers: list[int] = []
    for page in pages:
        stream_parts = [b"BT\n"]
        for text, x, y, size in page:
            stream_parts.append(
                f"/F1 {size} Tf\n1 0 0 1 {x} {y} Tm\n({_escape(text)}) Tj\n".encode()
            )
        stream_parts.append(b"ET")
        stream = b"".join(stream_parts)

        content_number = add(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
        page_number = add(
            b"<< /Type /Page /Parent "
            + str(pages_number).encode()
            + b" 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 "
            + str(font_number).encode()
            + b" 0 R >> >> /Contents "
            + str(content_number).encode()
            + b" 0 R >>"
        )
        page_numbers.append(page_number)

    kids = b" ".join(f"{number} 0 R".encode() for number in page_numbers)
    objects[pages_number - 1] = (
        b"<< /Type /Pages /Count " + str(len(page_numbers)).encode() + b" /Kids [" + kids + b"] >>"
    )
    objects[catalog_number - 1] = (
        b"<< /Type /Catalog /Pages " + str(pages_number).encode() + b" 0 R >>"
    )

    # Assemble the file, recording each object's byte offset for the xref table.
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(number).encode() + b" 0 obj\n" + body + b"\nendobj\n"

    xref_offset = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n"
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode()
        + b" /Root "
        + str(catalog_number).encode()
        + b" 0 R >>\nstartxref\n"
        + str(xref_offset).encode()
        + b"\n%%EOF\n"
    )
    return bytes(out)


def build_image_only_pdf(page_count: int = 2) -> bytes:
    """Build a PDF whose pages contain no text at all.

    Stands in for a scanned document: the parser must recognise that there is nothing to
    extract and say so, rather than returning an empty result that looks like success.
    """
    objects: list[bytes] = [b"", b"", b""]
    font_number, catalog_number, pages_number = 1, 2, 3
    objects[font_number - 1] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    page_numbers: list[int] = []
    for _ in range(page_count):
        # A filled rectangle: real page content, but not a single glyph.
        stream = b"0.5 0.5 0.5 rg\n100 100 400 600 re\nf"
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
        content_number = len(objects)
        objects.append(
            b"<< /Type /Page /Parent "
            + str(pages_number).encode()
            + b" 0 R /MediaBox [0 0 612 792] /Resources << >> /Contents "
            + str(content_number).encode()
            + b" 0 R >>"
        )
        page_numbers.append(len(objects))

    kids = b" ".join(f"{number} 0 R".encode() for number in page_numbers)
    objects[pages_number - 1] = (
        b"<< /Type /Pages /Count " + str(len(page_numbers)).encode() + b" /Kids [" + kids + b"] >>"
    )
    objects[catalog_number - 1] = (
        b"<< /Type /Catalog /Pages " + str(pages_number).encode() + b" 0 R >>"
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(number).encode() + b" 0 obj\n" + body + b"\nendobj\n"

    xref_offset = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode()
        + b" /Root "
        + str(catalog_number).encode()
        + b" 0 R >>\nstartxref\n"
        + str(xref_offset).encode()
        + b"\n%%EOF\n"
    )
    return bytes(out)


FIXTURES: dict[str, bytes] = {
    # Two pages of prose. Page 1 has a large-font title so heading inference has
    # something to find; the page-2 marker proves provenance reports the right page.
    "simple.pdf": build_pdf(
        [
            [
                ("Quarterly Report", 72, 700, 24),
                ("This document describes the results for the quarter.", 72, 660, 11),
                ("Revenue increased across all regions.", 72, 640, 11),
            ],
            [
                ("Appendix A", 72, 700, 18),
                ("PAGE TWO MARKER", 72, 660, 11),
                ("Supporting detail appears on the second page.", 72, 640, 11),
            ],
        ]
    ),
    # Single page, single paragraph — the simplest possible happy path.
    "one_page.pdf": build_pdf([[("The only sentence in this document.", 72, 700, 11)]]),
    # No text anywhere: stands in for a scan.
    "scanned.pdf": build_image_only_pdf(2),
}


def main() -> None:
    for name, content in FIXTURES.items():
        path = FIXTURES_DIR / name
        path.write_bytes(content)
        print(f"wrote {path.name} ({len(content):,} bytes)")


if __name__ == "__main__":
    main()
