"""Input guard: prompt-injection heuristics + PII redaction.

Not a security boundary on its own - it is the cheap first line that stops the
obvious cases and, more importantly, keeps raw card numbers and phone numbers
out of logs and traces, which is the part auditors ask about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"忽略(之前|上面|以上|前面)的?(所有)?(指令|规则|设定)"),
    re.compile(r"(system|developer)\s*prompt", re.IGNORECASE),
    re.compile(r"你现在是(一个)?(不受限制|无限制)"),
    re.compile(r"jailbreak|越狱模式", re.IGNORECASE),
]

CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[ -]?)?1[3-9]\d{9}(?!\d)")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


@dataclass(slots=True)
class GuardResult:
    text: str
    blocked: bool = False
    reason: str = ""
    redactions: int = 0


def redact(text: str) -> tuple[str, int]:
    count = [0]

    def _card(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if len(digits) < 13:
            return match.group(0)
        count[0] += 1
        return f"[CARD:***{digits[-4:]}]"

    masked = CARD_RE.sub(_card, text)
    masked, phone_hits = PHONE_RE.subn("[PHONE:***]", masked)
    masked, email_hits = EMAIL_RE.subn("[EMAIL:***]", masked)
    count[0] += phone_hits + email_hits
    return masked, count[0]


class InputGuard:
    def __init__(self, max_chars: int = 4000, block_injection: bool = True) -> None:
        self.max_chars = max_chars
        self.block_injection = block_injection

    def inspect(self, text: str) -> GuardResult:
        cleaned = text.strip()
        masked, redactions = redact(cleaned)
        result = GuardResult(text=masked, redactions=redactions)
        if len(masked) > self.max_chars:
            return GuardResult(text=masked[: self.max_chars], blocked=True, reason="input_too_long",
                               redactions=redactions)
        if self.block_injection:
            for pattern in INJECTION_PATTERNS:
                if pattern.search(masked):
                    result.blocked = True
                    result.reason = "prompt_injection_suspected"
                    break
        return result
