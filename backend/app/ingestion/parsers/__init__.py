"""One parser per supported file format."""

from app.ingestion.parsers.base import DocumentParser, ParserLimits

__all__ = ["DocumentParser", "ParserLimits"]
