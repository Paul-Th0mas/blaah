from __future__ import annotations

import difflib
import re
import time

FUZZY_MIN_LEN = 4
FUZZY_RATIO = 0.85


class NameDetector:
    """Fires when one of the user's names appears in text, with a cooldown.

    Exact whole-word matching first; then a fuzzy pass so "Pauls" or "Paulo"
    from a transcription slip still count. Short names are exact-only to
    avoid false positives on common words.

    Introductions don't count: "Hi all, Paul here", "this is Paul from platform", "I'm Paul"
    are someone saying who they are (usually you, as a call starts), not someone asking you.
    """

    def __init__(self, names: list[str], cooldown_seconds: float):
        self.names = [n.strip() for n in names if n.strip()]
        self.cooldown = cooldown_seconds
        alt = "|".join(re.escape(n) for n in self.names)
        self._exact = re.compile(r"\b(?:" + alt + r")\b", re.IGNORECASE) if self.names else None
        self._intro = re.compile(
            rf"\bmy name is\s+(?:{alt})\b"
            rf"|\b(?:i'm|i am)\s+(?:{alt})\b(?!')"  # not "I'm Paul's backup"
            rf"|\bthis is\s+(?:{alt})(?=\s*(?:[,.!]|$|\s+(?:from|here|speaking|with|on)\b))"
            rf"|(?:^|(?<=[,.!?]))\s*(?:{alt})\s+(?:here|speaking)\b(?!')",  # not "Paul here's the thing"
            re.IGNORECASE,
        ) if self.names else None
        self._fuzzy = [n.lower() for n in self.names if len(n) >= FUZZY_MIN_LEN]
        self.last_fired = 0.0

    def check(self, text: str) -> str | None:
        """Return the matched word if a name is present and the cooldown has passed."""
        if not self._exact or not text or time.time() - self.last_fired < self.cooldown:
            return None
        hit = self._match(self._intro.sub(" ", text))
        if hit:
            self.last_fired = time.time()
        return hit

    def count(self, text: str) -> int:
        """How many mentions of the user's names the text holds (exact or fuzzy); no cooldown."""
        if not self._exact or not text:
            return 0
        text = self._intro.sub(" ", text)
        fuzzy = sum(1 for word in re.findall(r"[A-Za-z']+", text) if not self._exact.fullmatch(word) and self._sounds_like(word))
        return len(self._exact.findall(text)) + fuzzy

    def _match(self, text: str) -> str | None:
        m = self._exact.search(text)
        if m:
            return m.group(0)
        return next((word for word in re.findall(r"[A-Za-z']+", text) if self._sounds_like(word)), None)

    def _sounds_like(self, word: str) -> bool:
        w = word.lower()
        return any(difflib.SequenceMatcher(None, w, name).ratio() >= FUZZY_RATIO for name in self._fuzzy)
