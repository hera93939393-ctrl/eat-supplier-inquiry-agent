"""context.py — 카테고리별 근거 문서를 읽어서 프롬프트에 넣을 형태로 조립한다.

카테고리 하나당 문서 파일 하나로 이미 나뉘어 있으므로(docs/*.md), 여기서는 전체 문서를
그대로 근거로 쓴다 — 문서 자체가 카테고리별로 쪼개져 있어 "필요한 부분만" 조건을 만족한다.
"""
from pathlib import Path

DOCS_DIR = Path(__file__).parent / "docs"

CATEGORY_DOC_PATHS = {
    "document_review": DOCS_DIR / "document_review.md",
    "site_inspection": DOCS_DIR / "site_inspection.md",
    "procedure_timeline": DOCS_DIR / "procedure_timeline.md",
    "eligibility_restriction": DOCS_DIR / "eligibility_restriction.md",
}

_cache: dict[str, str] = {}


def assemble_context(category: str) -> str:
    """카테고리에 해당하는 근거 문서 전체를 반환한다. out_of_scope는 근거 문서가 없다."""
    if category not in CATEGORY_DOC_PATHS:
        return ""
    if category not in _cache:
        _cache[category] = CATEGORY_DOC_PATHS[category].read_text(encoding="utf-8")
    return _cache[category]


def _selftest_assemble_context():
    for category, path in CATEGORY_DOC_PATHS.items():
        text = assemble_context(category)
        assert text, f"{category} 문서가 비어있음"
        assert path.stem in str(path)
        print(f"assemble_context 통과 ({category}): {len(text)}자")

    empty = assemble_context("out_of_scope")
    assert empty == ""
    print("assemble_context 통과 (out_of_scope는 빈 문자열):", repr(empty))


if __name__ == "__main__":
    _selftest_assemble_context()
