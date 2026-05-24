from __future__ import annotations

from agent_service.rag.documents import KnowledgeChunk, KnowledgeDocument

SEPARATORS = ("\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "，", ",", " ")


class RecursiveTextChunker:
    def __init__(self, *, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def split_documents(self, documents: list[KnowledgeDocument]) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        for document in documents:
            pieces = self.split_text(document.text)
            for index, piece in enumerate(pieces):
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=f"{document.doc_id}:{index:04d}",
                        doc_id=document.doc_id,
                        source=document.source,
                        title=document.title,
                        text=piece,
                        active=document.active,
                    )
                )
        return chunks

    def split_text(self, text: str) -> list[str]:
        units = _split_recursive(text.strip(), 0, self._chunk_size)
        chunks: list[str] = []
        current = ""
        for unit in units:
            candidate = unit if not current else f"{current} {unit}"
            if len(candidate) <= self._chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
            current = unit
        if current:
            chunks.append(current)

        if self._chunk_overlap == 0 or len(chunks) <= 1:
            return chunks
        return _apply_overlap(chunks, self._chunk_overlap, self._chunk_size)


def _split_recursive(text: str, separator_index: int, chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text else []
    if separator_index >= len(SEPARATORS):
        return [
            text[start : start + chunk_size]
            for start in range(0, len(text), chunk_size)
            if text[start : start + chunk_size].strip()
        ]

    separator = SEPARATORS[separator_index]
    parts = [part.strip() for part in text.split(separator) if part.strip()]
    if len(parts) <= 1:
        return _split_recursive(text, separator_index + 1, chunk_size)

    units: list[str] = []
    for part in parts:
        if len(part) <= chunk_size:
            units.append(part)
        else:
            units.extend(_split_recursive(part, separator_index + 1, chunk_size))
    return units


def _apply_overlap(chunks: list[str], overlap: int, chunk_size: int) -> list[str]:
    overlapped = [chunks[0]]
    for chunk in chunks[1:]:
        prefix = overlapped[-1][-overlap:].strip()
        combined = f"{prefix} {chunk}".strip() if prefix else chunk
        overlapped.append(combined[-chunk_size:] if len(combined) > chunk_size else combined)
    return overlapped
