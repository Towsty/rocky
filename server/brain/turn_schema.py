"""Deskbot turn contract: what one model turn returns.

speak  — 1–3 short Rocky sentences; only TTS text (no markdown/lists/emoji)
show   — optional screen payload; never spoken
emotion — neutral|happy|sad|angry|surprised|sleepy|thinking
acts   — tool calls taken before/during speak (look, track_face, …)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    speak: str
    emotion: str
    show: str | None = None
    acts: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        out = {
            "emotion": self.emotion,
            "speak": self.speak,
            "show": self.show,
            "acts": list(self.acts),
        }
        return out
