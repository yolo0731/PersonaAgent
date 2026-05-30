from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from agent_service.rag.documents import RetrievalTrace
from agent_service.style.filters import is_learnable_style_text, normalize_style_text


class StyleDialoguePair(BaseModel):
    pair_id: str = Field(min_length=1)
    persona_id: str = Field(min_length=1)
    self_speaker: str = Field(min_length=1)
    target_speaker: str = Field(min_length=1)
    self_text: str = Field(min_length=1)
    target_reply: str = Field(min_length=1)
    timestamp_ms: int = Field(ge=0)
    source_image: str = ""


class RetrievedStylePair(BaseModel):
    pair: StyleDialoguePair
    score: float


class StylePairRetrieval(BaseModel):
    results: list[RetrievedStylePair]
    trace: RetrievalTrace


class StylePairStore:
    def __init__(self, pairs: Sequence[StyleDialoguePair]) -> None:
        self._pairs = list(pairs)
        self.collection_name = "style_pair"

    @classmethod
    def from_jsonl(cls, path: str | Path) -> StylePairStore:
        source = Path(path)
        if not source.exists():
            return cls([])
        pairs: list[StyleDialoguePair] = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                pair = StyleDialoguePair.model_validate(raw)
            except (json.JSONDecodeError, ValidationError):
                continue
            if not _is_learnable_pair(pair):
                continue
            pairs.append(pair)
        return cls(pairs)

    def retrieve_pairs(
        self,
        *,
        persona_id: str,
        query: str,
        top_k: int = 3,
    ) -> StylePairRetrieval:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scored = [
            RetrievedStylePair(pair=pair, score=_score_pair(query, pair))
            for pair in self._pairs
            if pair.persona_id == persona_id
        ]
        ranked = sorted(
            (item for item in scored if item.score > 0),
            key=lambda item: (-item.score, -item.pair.timestamp_ms, item.pair.pair_id),
        )[:top_k]
        return StylePairRetrieval(
            results=ranked,
            trace=RetrievalTrace(
                query=query,
                top_k=top_k,
                result_count=len(ranked),
                collection=self.collection_name,
                chunk_ids=[item.pair.pair_id for item in ranked],
            ),
        )


def _is_learnable_pair(pair: StyleDialoguePair) -> bool:
    return is_learnable_style_text(pair.self_text) and is_learnable_style_text(pair.target_reply)


def _score_pair(query: str, pair: StyleDialoguePair) -> float:
    query_text = _compact(query)
    self_text = _compact(pair.self_text)
    target_text = _compact(pair.target_reply)
    if not query_text or not self_text:
        return 0.0

    score = _overlap_score(query_text, self_text)
    score += 0.3 * _overlap_score(query_text, target_text)
    score += 1.0 * _ngram_overlap_score(query_text, self_text, size=2)
    score += 1.5 * _ngram_overlap_score(query_text, self_text, size=3)
    if self_text in query_text or query_text in self_text:
        score += 3.0
    if _longest_common_substring_len(query_text, self_text) >= 4:
        score += 1.0
    return score


def _overlap_score(query_text: str, candidate_text: str) -> float:
    query_chars = set(query_text)
    candidate_chars = set(candidate_text)
    if not query_chars or not candidate_chars:
        return 0.0
    overlap = len(query_chars & candidate_chars)
    return overlap / math.sqrt(len(query_chars) * len(candidate_chars))


def _ngram_overlap_score(query_text: str, candidate_text: str, *, size: int) -> float:
    query_ngrams = _meaningful_ngrams(query_text, size=size)
    candidate_ngrams = _meaningful_ngrams(candidate_text, size=size)
    if not query_ngrams or not candidate_ngrams:
        return 0.0
    overlap = len(query_ngrams & candidate_ngrams)
    return overlap / math.sqrt(len(query_ngrams) * len(candidate_ngrams))


def _longest_common_substring_len(left: str, right: str) -> int:
    longest = 0
    previous = [0] * (len(right) + 1)
    for left_char in left:
        current = [0]
        for index, right_char in enumerate(right, start=1):
            value = previous[index - 1] + 1 if left_char == right_char else 0
            longest = max(longest, value)
            current.append(value)
        previous = current
    return longest


def _compact(text: str) -> str:
    return "".join(_meaningful_chars(normalize_style_text(text).casefold()))


_COMMON_NGRAMS = {
    "我马",
    "马上",
    "有点",
    "这个",
    "就是",
    "现在",
    "今天",
    "明天",
}


def _meaningful_ngrams(text: str, *, size: int) -> set[str]:
    if len(text) < size:
        return set()
    return {
        text[index : index + size]
        for index in range(0, len(text) - size + 1)
        if text[index : index + size] not in _COMMON_NGRAMS
        and not text[index : index + size].startswith(("我", "你"))
    }


def _meaningful_chars(text: str) -> Iterable[str]:
    for match in re.finditer(r"[\u4e00-\u9fffA-Za-z0-9]", text):
        yield match.group(0)
