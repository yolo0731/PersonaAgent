from __future__ import annotations

from pathlib import Path

from agent_service.rag.documents import KnowledgeDocument

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt"}


class DocumentLoader:
    def load_directory(self, directory: str | Path) -> list[KnowledgeDocument]:
        root = Path(directory)
        documents: list[KnowledgeDocument] = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                documents.append(self.load_file(path, root=root))
        return documents

    def load_file(self, path: str | Path, *, root: str | Path | None = None) -> KnowledgeDocument:
        file_path = Path(path)
        text = file_path.read_text(encoding="utf-8")
        source = str(file_path if root is None else file_path.relative_to(Path(root)))
        return KnowledgeDocument(
            doc_id=_make_doc_id(file_path if root is None else file_path.relative_to(Path(root))),
            source=source,
            title=_extract_title(text, fallback=file_path.stem),
            text=text,
            active=True,
        )


def _make_doc_id(path: Path) -> str:
    return path.with_suffix("").as_posix().replace("/", "__")


def _extract_title(text: str, *, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
        return stripped[:80]
    return fallback
