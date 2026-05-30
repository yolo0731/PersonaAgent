from __future__ import annotations

import re

UI_DATE_TEXT_PATTERN = re.compile(
    r"^(?:星期[一二三四五六日天]|周[一二三四五六日天]|今天|昨天|前天|"
    r"\d{1,2}:\d{2}|"
    r"\d{1,2}/\d{1,2}/\d{2,4}|"
    r"\d{2,4}[-/年]\d{1,2}[-/月]\d{1,2}日?(?:\s+\d{1,2}:\d{2})?|"
    r"\d+)$"
)

NON_TEXT_MESSAGES = {
    "[图片]",
    "[表情]",
    "[动画表情]",
    "[视频]",
    "[语音]",
    "[文件]",
    "[位置]",
    "[链接]",
}
GENERATED_SAFETY_TEXTS = {
    "我会保持相近的简洁语气，但不会复述授权样本原文。",
}

_MONEY_NOISE_PATTERN = re.compile(
    r"^[￥¥$]\s*\d[\d.,]*(?:\s*[*xX×+\-])?$"
)
_SHORT_OCR_NOISE_PATTERN = re.compile(r"^(?:加[-—_]?|[-—_]+|[+*#]+)$")
_SYMBOL_ONLY_PATTERN = re.compile(
    r"^[\s￥¥$€£\d.,，。:：;；/\\|*×xX+\-=_%（）()\[\]【】<>《》~～]+$"
)
_CJK_OR_ALNUM_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
_WEEKDAY_FRAGMENT_PATTERN = re.compile(r"^期[一二三四五六日天]$")
_ALLOWED_SINGLE_CHAR_REPLIES = {"嗯", "恩", "哦", "噢", "喔", "啊", "好", "行", "困", "累", "饿"}
_PROMPT_EXAMPLE_BLOCKLIST = {"点击", "见侍", "具头", "上父"}
_PROMPT_EXAMPLE_PUNCTUATION = set("，。！？!?～~…")
_CODE_LIKE_PATTERN = re.compile(
    r"(?:"
    r"\b(?:auto|bool|class|def|for|if|import|int|private|protected|public|return|std::|while)\b|"
    r"//|/\*|\*/|;|"
    r"[A-Za-z_][A-Za-z0-9_]*\s*[\[.]|"
    r"[A-Za-z_][A-Za-z0-9_]*\s*(?:\+=|-=|\+\+|--|=)"
    r")"
)


def normalize_style_text(text: str) -> str:
    return " ".join(text.strip().split())


def is_learnable_style_text(text: str) -> bool:
    cleaned = normalize_style_text(text)
    if not cleaned:
        return False
    if len(cleaned) > 120:
        return False
    if cleaned in NON_TEXT_MESSAGES:
        return False
    if cleaned in GENERATED_SAFETY_TEXTS:
        return False
    if cleaned.startswith("mock reply:"):
        return False
    if UI_DATE_TEXT_PATTERN.match(cleaned):
        return False
    if _MONEY_NOISE_PATTERN.match(cleaned):
        return False
    if _SHORT_OCR_NOISE_PATTERN.match(cleaned):
        return False
    if _WEEKDAY_FRAGMENT_PATTERN.match(cleaned):
        return False
    if _CODE_LIKE_PATTERN.search(cleaned):
        return False
    if _SYMBOL_ONLY_PATTERN.match(cleaned):
        return False
    if _CJK_OR_ALNUM_PATTERN.search(cleaned) is None:
        return False
    if len(cleaned) == 1 and cleaned not in _ALLOWED_SINGLE_CHAR_REPLIES:
        return False
    return True


def is_prompt_style_example(text: str) -> bool:
    cleaned = normalize_style_text(text)
    if not is_learnable_style_text(cleaned):
        return False
    if any(fragment in cleaned for fragment in _PROMPT_EXAMPLE_BLOCKLIST):
        return False
    if re.search(r"[A-Za-z0-9]", cleaned):
        return False
    if len(cleaned) > 24 and not any(char in _PROMPT_EXAMPLE_PUNCTUATION for char in cleaned):
        return False
    return True
