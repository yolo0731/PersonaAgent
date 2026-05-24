from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, Field

from agent_service.governance.data_manifest import ProcessedStyleSample

EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF]")
CONTENT_CHAR_PATTERN = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
SENTENCE_SPLIT_PATTERN = re.compile(r"[。！？!?]+")
PUNCTUATION_CHARS = set("。！？!?，,；;：:、～~…")
DEFAULT_TONE_PARTICLES = ("呀", "啦", "呢", "吧", "哦", "嘛", "哈")
DEFAULT_CATCHPHRASES = ("好呀", "收到", "没问题", "马上安排", "哈哈", "嗯嗯")


class ReplyLengthDistribution(BaseModel):
    min_chars: int = Field(ge=0)
    max_chars: int = Field(ge=0)
    average_chars: float = Field(ge=0.0)
    short_count: int = Field(ge=0)
    medium_count: int = Field(ge=0)
    long_count: int = Field(ge=0)


class StyleFeatures(BaseModel):
    sample_count: int = Field(ge=0)
    average_sentence_length: float = Field(ge=0.0)
    punctuation_counts: dict[str, int]
    emoji_count: int = Field(ge=0)
    emoji_per_sample: float = Field(ge=0.0)
    tone_particle_counts: dict[str, int]
    catchphrase_counts: dict[str, int]
    reply_length_distribution: ReplyLengthDistribution


class StyleFeatureExtractor:
    def __init__(
        self,
        *,
        tone_particles: Sequence[str] = DEFAULT_TONE_PARTICLES,
        catchphrase_candidates: Sequence[str] = DEFAULT_CATCHPHRASES,
    ) -> None:
        self._tone_particles = tuple(tone_particles)
        self._catchphrase_candidates = tuple(catchphrase_candidates)

    def extract(self, samples: Sequence[ProcessedStyleSample]) -> StyleFeatures:
        texts = [sample.text for sample in samples]
        sample_count = len(texts)
        sentence_lengths = _sentence_lengths(texts)
        reply_lengths = [len(text) for text in texts]
        punctuation_counts: Counter[str] = Counter()
        tone_counts: Counter[str] = Counter()
        catchphrase_counts: Counter[str] = Counter()
        emoji_count = 0

        for text in texts:
            punctuation_counts.update(char for char in text if char in PUNCTUATION_CHARS)
            emoji_count += len(EMOJI_PATTERN.findall(text))
            for particle in self._tone_particles:
                count = text.count(particle)
                if count > 0:
                    tone_counts[particle] += count
            for phrase in self._catchphrase_candidates:
                count = text.count(phrase)
                if count > 0:
                    catchphrase_counts[phrase] += count

        return StyleFeatures(
            sample_count=sample_count,
            average_sentence_length=_average(sentence_lengths),
            punctuation_counts=dict(punctuation_counts),
            emoji_count=emoji_count,
            emoji_per_sample=_average_per_sample(emoji_count, sample_count),
            tone_particle_counts=dict(tone_counts),
            catchphrase_counts=dict(catchphrase_counts),
            reply_length_distribution=_reply_length_distribution(reply_lengths),
        )


def _sentence_lengths(texts: Sequence[str]) -> list[int]:
    lengths: list[int] = []
    for text in texts:
        for sentence in SENTENCE_SPLIT_PATTERN.split(text):
            content_length = len(CONTENT_CHAR_PATTERN.findall(sentence))
            if content_length > 0:
                lengths.append(content_length)
    return lengths


def _reply_length_distribution(lengths: Sequence[int]) -> ReplyLengthDistribution:
    if not lengths:
        return ReplyLengthDistribution(
            min_chars=0,
            max_chars=0,
            average_chars=0.0,
            short_count=0,
            medium_count=0,
            long_count=0,
        )
    return ReplyLengthDistribution(
        min_chars=min(lengths),
        max_chars=max(lengths),
        average_chars=_average(lengths),
        short_count=sum(1 for length in lengths if length <= 20),
        medium_count=sum(1 for length in lengths if 20 < length <= 80),
        long_count=sum(1 for length in lengths if length > 80),
    )


def _average(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def _average_per_sample(total: int, sample_count: int) -> float:
    if sample_count == 0:
        return 0.0
    return round(total / sample_count, 2)
