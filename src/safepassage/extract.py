"""Entity extraction: category, language, country, agency, fee, salary. Owner: Person A."""

from .schema import Content


def extract_content(text: str) -> Content:
    # TODO(A): category classifier or keywords, language/script detection, country + agency + fee extraction
    return Content()
