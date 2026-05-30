from __future__ import annotations


def parse_remember_content(text: str) -> str | None:
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered.startswith("/remember"):
        content = stripped[len("/remember") :].strip()
        return content or None
    if stripped.startswith("记住"):
        content = stripped[len("记住") :].strip(" ：:")
        return content or None
    for suffix in ("，记住", ",记住", " 记住", "记住"):
        if stripped.endswith(suffix) and len(stripped) > len(suffix):
            content = stripped[: -len(suffix)].strip(" ，,:：")
            return content or None
    return None


def parse_forget_memory_id(text: str) -> str | None:
    parts = text.strip().split(maxsplit=1)
    if len(parts) != 2:
        return None
    if parts[0].lower() != "/forget":
        return None
    return parts[1].strip() or None
