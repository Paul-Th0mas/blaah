"""Who said what: voice clustering plus name cues.

Every finalised sentence gets a voiceprint (TitaNet-small via sherpa-onnx, ~25 ms on the
CPU). Voiceprints are clustered online into voices S1, S2, … Names are attached to voices
from what people say:

    "Dev, start with the cutover."   → the next *different* voice is Dev      (address)
    "Hi all, Raj here." / "I'm Sarah" → this voice is Raj / Sarah               (self)
    "Thanks, Karen."                  → the previous different voice is Karen  (thanks)
    whoever addresses Dev is not Dev                                           (negative)

Evidence accumulates per voice and names are assigned one-to-one, so a single wrong cue
("Karen, hold the comms" followed by someone else) is outvoted. Optional extra evidence:
an enrolled voiceprint for the user, remembered colleagues' voiceprints, and — in calls
captured with system audio — whether the sentence came from the mic or the speakers.

Calibrated on a real four-person meeting (AMI ES2004a, all crosstalk kept): 4 voices
found, 90% of speech time in the right voice. Sentences shorter than ~2 s carry too little
voice to cluster reliably, so they never start a new voice.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import quote, unquote

import numpy as np

from .config import PROJECT_ROOT, SpeakersCfg

MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx"

MIN_EMBED_SECONDS = 0.5  # below this no voiceprint at all
FOUND_SECONDS = 2.0  # only sentences at least this long may start a new voice
SHORT_MARGIN = 0.10  # a short sentence joins its best voice only if clearly ahead of the next
MERGE_SIMILARITY = 0.60  # two voices this alike are one person split in two
PRINT_MATCH = 0.50  # similarity to a saved voiceprint that counts as a match

# Evidence weights
W_SELF, W_ADDRESS, W_THANKS, W_NOT, W_PRINT, W_SOURCE = 3.0, 2.0, 2.0, -2.0, 3.0, 1.0
MIN_SCORE = 2.0  # at least one solid cue before a voice gets a name
HANDOFF_MAX_SEGMENTS = 4  # an address cue waits at most this many sentences by the addresser

# Capitalised words that start or end sentences but are not names.
NOT_NAMES = set(
    """
    okay ok so and but now next then well yes yeah yep no nope sure great good fine right alright
    thanks thank cheers sorry hi hello hey morning afternoon evening everyone everybody all guys folks team
    agreed perfect cool nice exactly absolutely definitely totally honestly basically actually anyway also
    just maybe please wait hmm um uh oh ah look listen first last finally today tomorrow yesterday
    monday tuesday wednesday thursday friday saturday sunday january february march april may june july
    august september october november december q1 q2 q3 q4 i we you they he she it this that there here
    what why how when where who which yes indeed true sounds lovely brilliant awesome super fantastic
    noted done correct sure question questions update updates hold over go let let's
    a an the to of in on at by for with from as is are was were be been am do does did have has had
    not nor or if than too very can could will would shall should might must need yet still even
    only once both each every any some more most much many few less such same other another
    again ever never always often soon later early late away back down off out up into onto upon
    about above after before below between during until while since though although because unless
    one two three four five six seven eight nine ten half part side kind sort lot bit way time day
    week month year thing things point idea plan step work job role team call talk chat meeting
    deal fair true real sure fine cool nice glad happy sad bad best better worse worst big small
    new old long short high low full open close clear quick fast slow easy hard free busy ready
    yes yeah yup okay hmm huh wow ooh oops ugh shh mhm uh-huh
    see say said tell told ask asked know knew think thought feel felt mean meant want wanted
    make made take took give gave get got put keep kept come came show try tried use used
    move send sent read run ran pay paid set start stop turn leave left call called
    """.split()
)
MIN_DICTIONARY_CHECK = 5  # shorter words collide with rare dictionary entries (dev, raj, sam)

_NAME = r"([A-Z][a-z]+(?:-[A-Z][a-z]+)?)"
_LEAD = r"^(?:(?:and|so|okay|ok|right|great|good|thanks|thank you|alright|now|next|well|yes|no|sure|perfect)[,.!]?\s+)*"
RE_ADDRESS_START = re.compile(_LEAD + _NAME + r"\s*[,?!]", re.IGNORECASE)
RE_ADDRESS_END = re.compile(r",\s*" + _NAME + r"\s*[?.!]*$", re.IGNORECASE)
RE_FIRST_WORD = re.compile(_LEAD + r"([A-Za-z]+)\b", re.IGNORECASE)

# Sound-alike matching against the meeting's attendees. The speech model mangles names it
# hasn't learned ("Gavin" → "given", "Kivinus"; "Shaun" → "shown"), so in cue positions a
# word that *sounds like* an attendee counts as them. Only in cue positions: "they have
# given us" is never read as Gavin.
ATTENDEE_MATCH = 0.75  # sound-alike similarity for a word in a comma-marked cue position
ATTENDEE_STRICT = 0.80  # …for a bare sentence-initial word ("Kivinus I forgot to mention")
_SOUND_RULES = (("ph", "f"), ("sh", "x"), ("ch", "x"), ("ck", "k"), ("c", "k"), ("q", "k"),
                ("g", "k"), ("v", "f"), ("z", "s"), ("w", ""))


def sound_key(word: str) -> str:
    """Crude sound-alike key: similar consonants merged, vowels dropped after the first letter."""
    w = re.sub(r"[^a-z]", "", word.lower())
    for a, b in _SOUND_RULES:
        w = w.replace(a, b)
    return w[:1] + re.sub(r"[aeiouy]", "", w[1:])


def attendee_match(word: str | None, attendees: list[str] | tuple[str, ...], threshold: float = ATTENDEE_MATCH) -> str | None:
    """The attendee a (possibly misheard) word refers to, or None."""
    from difflib import SequenceMatcher

    if not word or not attendees:
        return None
    w = word.lower()
    for name in attendees:
        if w == name.lower():
            return name
    if len(w) < 3 or w in NOT_NAMES:
        return None
    key = sound_key(w)
    score, best = max((SequenceMatcher(None, key, sound_key(n)).ratio(), n) for n in attendees)
    return best if score >= threshold else None
RE_SELF = [
    re.compile(r"\bmy name is\s+" + _NAME, re.IGNORECASE),
    re.compile(r"\b(?:i'm|i am)\s+" + _NAME, re.IGNORECASE),
    re.compile(r"\bthis is\s+" + _NAME + r"(?=\s*(?:,|\.|!|$|\s+(?:from|here|speaking|with|on)\b))", re.IGNORECASE),
    re.compile(r"(?:^|,\s*)" + _NAME + r"\s+here\b", re.IGNORECASE),
]
RE_THANKS = re.compile(r"^(?:thanks|thank you|cheers)[,!]?\s+" + _NAME + r"\b", re.IGNORECASE)

_DICTIONARY = Path("/usr/share/dict/words")
_common_words: set[str] | None = None


def _everyday_words() -> set[str]:
    """Words the system dictionary lists only in lowercase — ordinary words, not names."""
    global _common_words
    if _common_words is None:
        lower, proper = set(), set()
        if _DICTIONARY.exists():
            for w in _DICTIONARY.read_text(errors="ignore").split():
                (proper if w[:1].isupper() else lower).add(w.lower())
        _common_words = lower - proper  # "Mark", "Grace", "Will" stay possible names
    return _common_words


def _stems(w: str) -> set[str]:
    """The word plus plausible base forms (hiring → hire, approved → approve, replicas → replica)."""
    out = {w}
    for suffix, adds in (("ies", ("y",)), ("ing", ("", "e")), ("ed", ("", "e")), ("es", ("",)), ("s", ("",)), ("ly", ("",))):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            base = w[: -len(suffix)]
            out |= {base + a for a in adds}
            if len(base) > 3 and base[-1] == base[-2]:  # running → run, planned → plan
                out.add(base[:-1])
    return out


def _is_name(word: str | None, known: frozenset[str] | set[str] = frozenset(), proper: frozenset[str] | set[str] = frozenset()) -> bool:
    """`known`: the user and configured people (always names). `proper`: words this meeting
    has capitalised mid-sentence — enough to outweigh the dictionary, not the stoplist."""
    if not word or len(word) < 2 or not word[0].isupper():
        return False
    w = word.lower()
    if w in known:
        return True
    if w in NOT_NAMES:
        return False
    if len(w) >= MIN_DICTIONARY_CHECK and w not in proper and _stems(w) & _everyday_words():
        return False
    return True


def extract_cues(
    text: str,
    known: frozenset[str] | set[str] = frozenset(),
    proper: frozenset[str] | set[str] = frozenset(),
    attendees: list[str] | tuple[str, ...] = (),
) -> list[tuple[str, str]]:
    """[(kind, name)] with kind in {"self", "address", "thanks"} for one sentence.

    A word in a cue position counts as a name if it sounds like an attendee (returned as the
    attendee's spelling), or otherwise looks like a name (see _is_name).
    """
    text = text.strip()

    def name_of(word: str | None) -> str | None:
        if not word or word.lower() in {"thanks", "thank"}:
            return None
        return attendee_match(word, attendees) or (word if _is_name(word, known, proper) else None)

    cues: list[tuple[str, str]] = []
    for pattern in RE_SELF:
        for m in pattern.finditer(text):
            name = name_of(m.group(1))
            if name and ("self", name) not in cues:
                cues.append(("self", name))
    m = RE_THANKS.match(text)
    if m and name_of(m.group(1)):
        cues.append(("thanks", name_of(m.group(1))))
    elif not cues:  # "I'm Sarah, …" is a self cue, not an address
        m = RE_ADDRESS_START.match(text) or RE_ADDRESS_END.search(text)
        name = name_of(m.group(1)) if m else None
        if not name and attendees:
            name = _attendee_address(text, attendees)
        if name:
            cues.append(("address", name))
    return cues


RE_COMMA_WORD = re.compile(r",\s*([A-Za-z]+)\b(?:\s+([A-Za-z']+))?")
# After ", Paul" these make it a mention, not an address: ", Paul was on it", ", Paul and I".
THIRD_PERSON_NEXT = set("is was has had said says will would did does can could should mentioned thinks thought "
                        "and or who which that from also too as s's".split())
# Before a name these make it a mention: "with Gavin", "ask Gavin".
MENTION_BEFORE = set("with to about for from and or of by tell ask told asked like than cc ping at on in".split())
YOU_WITHIN = 7  # words after a name in which "you" marks it as an address


def _plain_word(word: str) -> bool:
    return not (_stems(word.lower()) & _everyday_words()) and word.lower() not in NOT_NAMES


def _attendee_address(text: str, attendees: list[str] | tuple[str, ...]) -> str | None:
    """Address cues that need the attendee list, for unpunctuated or misheard names."""
    # ", Paul sorry…" / ", Gavin." — a name after a comma, not followed by a third-person verb.
    for m in RE_COMMA_WORD.finditer(text):
        nxt = (m.group(2) or "").lower()
        name = attendee_match(m.group(1), attendees)
        if name and nxt not in THIRD_PERSON_NEXT:
            return name
    # "uh Kivinus I forgot to mention, did you…" — a strong, non-dictionary sound-alike of an
    # attendee followed shortly by "you". Ordinary words ("given") never qualify here.
    words = re.findall(r"[A-Za-z']+", text)
    for i, word in enumerate(words):
        if not _plain_word(word) or (i and words[i - 1].lower() in MENTION_BEFORE):
            continue
        name = attendee_match(word, attendees, ATTENDEE_STRICT)
        if name and any(w.lower() in ("you", "your") for w in words[i + 1 : i + 1 + YOU_WITHIN]):
            return name
    return None


BACKCHANNEL_WORDS = set(
    "okay ok cool sorry yeah yes yep yup sure right thanks thank you great good sounds nice perfect alright "
    "all uh um mm mhm hmm excellent there no oh fine got it bye see ya so well".split()
)


def is_backchannel(text: str) -> bool:
    """'Sorry.', 'Cool.', 'Sounds good, yeah.' — reactions, not a turn. Too short to place by voice,
    and rarely the answer a hand-off is waiting for, so name cues look past them."""
    words = re.findall(r"[a-z']+", text.lower())
    return not words or all(w in BACKCHANNEL_WORDS for w in words)


RE_MID_CAPITAL = re.compile(r"(?<=[a-z,;]\s)([A-Z][a-z]+)\b")


def proper_nouns(text: str) -> set[str]:
    """Lowercased words capitalised mid-sentence ("tag Arjun on it") — evidence they are names."""
    return {w.lower() for w in RE_MID_CAPITAL.findall(text)}


def ensure_model(path: str | Path) -> Path:
    p = Path(path).expanduser()
    p = p if p.is_absolute() else PROJECT_ROOT / p
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.rename(p)
    return p


class VoiceEmbedder:
    def __init__(self, model_path: str | Path, threads: int = 2):
        import sherpa_onnx

        self._ex = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(ensure_model(model_path)), num_threads=threads)
        )

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        stream = self._ex.create_stream()
        stream.accept_waveform(sample_rate, np.ascontiguousarray(audio, dtype=np.float32))
        stream.input_finished()
        v = np.asarray(self._ex.compute(stream), dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-9)


@dataclass
class Voice:
    id: str
    total: np.ndarray  # duration-weighted sum of voiceprints
    seconds: float = 0.0
    sentences: int = 0

    @property
    def centroid(self) -> np.ndarray:
        return self.total / (np.linalg.norm(self.total) + 1e-9)

    def add(self, emb: np.ndarray, seconds: float) -> None:
        self.total = self.total + emb * min(seconds, 10.0)
        self.seconds += seconds
        self.sentences += 1


@dataclass
class Handoff:
    name: str
    by: str
    waited: int = 0


@dataclass
class SpeakerTracker:
    """Voices and their names for one meeting.

    Used from the transcriber thread (every sentence) and the UI thread (attendees, hand
    naming, the status bar), so every public method holds one lock. `on_change` is called
    after the lock is released: it re-renders and may wait for the UI thread.
    """

    cfg: SpeakersCfg
    user_names: list[str]
    embedder: VoiceEmbedder | None = None
    voiceprints: dict[str, np.ndarray] = field(default_factory=dict)  # name -> unit vector
    sample_rate: int = 16000
    on_change: Callable[[dict[str, str]], None] = lambda _names: None

    def __post_init__(self) -> None:
        self.voices: dict[str, Voice] = {}
        self.names: dict[str, str] = {}  # voice id -> name (resolved)
        self.manual: dict[str, str] = {}  # voice id -> name set by hand; always wins
        self.attendees: list[str] = []
        # Every sentence as (voice id, text, mic/speaker source): lets cues be re-read when the
        # attendee list changes, and lets a finished meeting be relabelled.
        self.history: list[tuple[str, str, str | None]] = []
        self.last: str = ""  # voice of the latest sentence
        self._prev_turn: str = ""  # voice of the latest sentence that wasn't just a reaction
        self._cue_ev: dict[str, dict[str, float]] = {}  # from what was said + source; rebuildable
        self._print_ev: dict[str, dict[str, float]] = {}  # from saved voiceprints
        self._handoff: Handoff | None = None
        self._aliases = {n.lower(): (self.user_names[0] if self.user_names else n) for n in self.user_names}
        self._merged: dict[str, str] = {}
        self._relabel = False  # something changed what some lines display
        self.known: set[str] = {n.lower() for n in self.user_names} | {n.lower() for n in self.cfg.people}
        self.proper: set[str] = set()  # capitalised mid-sentence somewhere in this meeting
        self._lock = threading.RLock()
        self._changed: dict[str, str] | None = None  # names to report once the lock is released

    # ----- public -----
    def set_user_names(self, names: list[str]) -> None:
        with self._lock:
            self.user_names = list(names)
            self._aliases = {n.lower(): names[0] for n in names}
            self.known |= {n.lower() for n in names}
            self._replay()
        self._notify()

    @property
    def user(self) -> str:
        return self.user_names[0] if self.user_names else ""

    @property
    def evidence(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for store in (self._cue_ev, self._print_ev):
            for vid, ev in store.items():
                row = out.setdefault(vid, {})
                for name, w in ev.items():
                    row[name] = row.get(name, 0.0) + w
        return out

    def set_attendees(self, names: list[str]) -> None:
        """Who is in this meeting; re-reads everything said so far with the new names."""
        with self._lock:
            self.attendees = [n.strip() for n in names if n.strip()]
            self.known |= {n.lower() for n in self.attendees}
            self._replay()
        self._notify()

    def set_name(self, voice_id: str, name: str) -> None:
        """Name a voice by hand ('' clears it). Hand-set names beat all evidence."""
        with self._lock:
            vid = self.resolve(voice_id)
            name = self._canon(name.strip()) if name.strip() else ""
            for other, n in list(self.manual.items()):
                if n == name:
                    del self.manual[other]
            if name:
                self.manual[vid] = name
            else:
                self.manual.pop(vid, None)
            self._relabel = True
            self._resolve_names()
        self._notify()

    def counts(self) -> tuple[int, int]:
        """(voices, named voices), for a status bar."""
        with self._lock:
            return len(self.voices), sum(1 for v in self.voices if v in self.names)

    def voice_summary(self) -> list[tuple[str, str, int, str]]:
        """(voice id, shown name, sentences, latest sentence) for every voice, busiest first."""
        counts: dict[str, int] = {}
        latest: dict[str, str] = {}
        with self._lock:
            for vid, text, _ in self.history:
                vid = self.resolve(vid)
                counts[vid] = counts.get(vid, 0) + 1
                latest[vid] = text
            return sorted(((v, self.display(v), counts.get(v, 0), latest.get(v, "")) for v in self.voices), key=lambda r: -r[2])

    @classmethod
    def from_meeting(cls, cfg: SpeakersCfg, user_names: list[str], segments: list, saved: dict | None) -> "SpeakerTracker":
        """Rebuild the naming state of a finished meeting from its transcript and speakers.json."""
        t = cls(cfg, user_names)
        saved = saved or {}
        t._merged = dict(saved.get("_merged", {}))
        for seg in segments:
            vid = t.resolve(getattr(seg, "speaker", ""))
            if vid and vid not in t.voices:
                t.voices[vid] = Voice(vid, np.zeros(1))
            if vid:
                t.history.append((vid, seg.text, getattr(seg, "source", "") or None))
        for vid, info in saved.items():
            if not vid.startswith("_") and isinstance(info, dict) and info.get("manual"):
                t.manual[vid] = info.get("name", "")
        t.attendees = list(saved.get("_attendees", []))
        t.known |= {n.lower() for n in t.attendees}
        t._replay()
        t._changed = None  # nobody is listening yet
        return t

    def resolve(self, voice_id: str) -> str:
        """Follow merges: a voice that was folded into another reports the survivor."""
        while voice_id in self._merged:
            voice_id = self._merged[voice_id]
        return voice_id

    def display(self, voice_id: str) -> str:
        """'Dev', 'Paul' or 'Speaker 2' for a voice id; names pass through unchanged."""
        if not voice_id:
            return ""
        vid = self.resolve(voice_id)
        if vid in self.names:
            return self.names[vid]
        if vid in self.voices:
            return f"Speaker {vid[1:]}"
        return voice_id  # already a name (e.g. ground-truth transcripts)

    def assign(self, audio: np.ndarray | None, text: str, source: str | None = None) -> str:
        """Voice id for one finalised sentence; updates names from its cues."""
        seconds = 0.0 if audio is None else len(audio) / self.sample_rate
        emb = None
        if self.embedder is not None and seconds >= MIN_EMBED_SECONDS:
            emb = self.embedder.embed(audio, self.sample_rate)  # the slow part: outside the lock
        with self._lock:
            vid = self._cluster(emb, seconds)
            if vid:
                self.history.append((vid, text, source))
                self._apply_sentence(vid, text, source, prev=self._prev_turn)
                self._merge_similar()
                vid = self.resolve(vid)
                self.last = vid
                if not is_backchannel(text):
                    self._prev_turn = vid
                self._resolve_names()
        self._notify()
        return vid

    def snapshot(self) -> dict:
        """JSON-able state for speakers.json."""
        with self._lock:
            return self._snapshot()

    def _snapshot(self) -> dict:
        evidence = self.evidence
        return {
            vid: {
                "name": self.names.get(vid, ""),
                "manual": vid in self.manual,
                "seconds": round(v.seconds, 1),
                "sentences": v.sentences,
                "evidence": {k: round(s, 1) for k, s in sorted(evidence.get(vid, {}).items(), key=lambda kv: -kv[1])},
            }
            for vid, v in self.voices.items()
        } | {"_merged": dict(self._merged), "_attendees": list(self.attendees)}

    def named_prints(self, min_seconds: float = 10.0) -> dict[str, np.ndarray]:
        """Voiceprints of confidently named voices, for remembering between meetings."""
        out = {}
        with self._lock:
            evidence = self.evidence
            for vid, name in self.names.items():
                v = self.voices[vid]
                sure = vid in self.manual or evidence.get(vid, {}).get(name, 0) >= MIN_SCORE
                if v.seconds >= min_seconds and sure:
                    out[name] = v.centroid
        return out

    # ----- clustering -----
    def _cluster(self, emb: np.ndarray | None, seconds: float) -> str:
        if emb is None:
            return self.last  # too short to tell: same voice as the sentence before
        scored = sorted(((float(v.centroid @ emb), vid) for vid, v in self.voices.items()), reverse=True)
        best_sim, best = scored[0] if scored else (-1.0, "")
        runner_up = scored[1][0] if len(scored) > 1 else -1.0
        if best and best_sim >= self.cfg.threshold:
            vid = best
        elif seconds >= FOUND_SECONDS:
            vid = f"S{len(self.voices) + 1}"
            self.voices[vid] = Voice(vid, np.zeros_like(emb))
            self._check_prints(vid, emb)
        elif best and best_sim - runner_up >= SHORT_MARGIN:
            vid = best  # short, but clearly closest to one voice
        else:
            return self.last  # short and ambiguous: assume the speaker continues
        if seconds >= FOUND_SECONDS:  # only solid sentences shape a voice
            self.voices[vid].add(emb, seconds)
        elif self.voices[vid].sentences == 0:
            self.voices[vid].add(emb, seconds)
        return vid

    def _merge_similar(self) -> None:
        ids = list(self.voices)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                if a in self.voices and b in self.voices:
                    if float(self.voices[a].centroid @ self.voices[b].centroid) >= MERGE_SIMILARITY:
                        self._fold(b, into=a)

    def _fold(self, src: str, into: str) -> None:
        s, d = self.voices.pop(src), self.voices[into]
        d.total, d.seconds, d.sentences = d.total + s.total, d.seconds + s.seconds, d.sentences + s.sentences
        for store in (self._cue_ev, self._print_ev):
            for name, w in store.pop(src, {}).items():
                self._add(store, into, name, w)
        if src in self.manual and into not in self.manual:
            self.manual[into] = self.manual[src]
        self.manual.pop(src, None)
        self._merged[src] = into
        self._relabel = True
        if self.last == src:
            self.last = into
        if self._handoff and self._handoff.by == src:
            self._handoff.by = into

    def _check_prints(self, vid: str, emb: np.ndarray) -> None:
        for name, vp in self.voiceprints.items():
            if float(vp @ emb) >= PRINT_MATCH:
                self._add(self._print_ev, vid, name, W_PRINT)

    # ----- names -----
    def _canon(self, name: str) -> str:
        return self._aliases.get(name.lower(), name[:1].upper() + name[1:])

    def _add(self, store: dict[str, dict[str, float]], vid: str, name: str, weight: float) -> None:
        name = self._canon(name)
        ev = store.setdefault(vid, {})
        ev[name] = ev.get(name, 0.0) + weight

    def _apply_sentence(self, vid: str, text: str, source: str | None, prev: str) -> None:
        self._apply_cues(vid, text, prev)
        if source in ("me", "others") and self.user:
            self._add(self._cue_ev, vid, self.user, W_SOURCE if source == "me" else -W_SOURCE)

    def _replay(self) -> None:
        """Rebuild cue evidence from the whole history (attendees changed, or a saved meeting)."""
        self._cue_ev, self._handoff = {}, None
        prev = ""
        for vid, text, source in self.history:
            vid = self.resolve(vid)
            self._apply_sentence(vid, text, source, prev)
            if not is_backchannel(text):
                prev = vid
        self._relabel = True
        self._resolve_names()

    def _apply_cues(self, vid: str, text: str, prev: str) -> None:
        """`prev`: voice of the latest real turn before this sentence (reactions skipped)."""
        # A pending hand-off lands on the first different voice that takes a real turn;
        # a quick "Sorry." or "Cool." from someone else neither takes it nor uses it up.
        if self._handoff and not is_backchannel(text):
            if vid != self.resolve(self._handoff.by):
                self._add(self._cue_ev, vid, self._handoff.name, W_ADDRESS)
                self._handoff = None
            else:
                self._handoff.waited += 1
                if self._handoff.waited > HANDOFF_MAX_SEGMENTS:
                    self._handoff = None
        self.proper |= proper_nouns(text)
        people = self.attendees or self.user_names[:1]  # you are always in your own meeting
        for kind, name in extract_cues(text, self.known, self.proper, people):
            if kind == "self":
                self._add(self._cue_ev, vid, name, W_SELF)
            elif kind == "address":
                self._add(self._cue_ev, vid, name, W_NOT)
                self._handoff = Handoff(self._canon(name), by=vid)
            elif kind == "thanks":
                self._add(self._cue_ev, vid, name, W_NOT)
                if prev and prev != vid:
                    self._add(self._cue_ev, prev, name, W_THANKS)

    def _resolve_names(self) -> None:
        names: dict[str, str] = {}
        taken: set[str] = set()
        for vid, name in self.manual.items():  # by hand first, always
            if vid in self.voices and name:
                names[vid] = name
                taken.add(name)
        pairs = sorted(
            ((score, vid, name) for vid, ev in self.evidence.items() if vid in self.voices for name, score in ev.items()),
            reverse=True,
        )
        for score, vid, name in pairs:
            if score < MIN_SCORE:
                break
            if vid not in names and name not in taken:
                names[vid] = name
                taken.add(name)
        if names != self.names or self._relabel:
            self.names = names
            self._relabel = False
            self._changed = dict(names)

    def _notify(self) -> None:
        """Report a change of names, outside the lock."""
        with self._lock:
            changed, self._changed = self._changed, None
        if changed is not None:
            self.on_change(changed)


def load_namer(meeting_dir: Path) -> Callable[[str], str]:
    """Voice id -> display name for a saved meeting, from its speakers.json."""
    f = meeting_dir / "speakers.json"
    if not f.exists():
        return lambda speaker: speaker
    data = json.loads(f.read_text())
    merged = data.get("_merged", {})

    def namer(speaker: str) -> str:
        while speaker in merged:
            speaker = merged[speaker]
        info = data.get(speaker)
        if isinstance(info, dict):
            return info.get("name") or f"Speaker {speaker[1:]}"
        return speaker

    return namer


# ----- voiceprint files: <meetings_dir>/_voices/<Name>.npy -----
def voices_dir(meetings_dir: Path) -> Path:
    return meetings_dir / "_voices"


def voice_file(meetings_dir: Path, name: str) -> Path:
    """Where a person's voiceprint lives. The name is %-encoded, so "José" and "Mary Ann"
    survive the trip to a file name and back."""
    return voices_dir(meetings_dir) / f"{quote(name, safe='')}.npy"


def load_voiceprints(meetings_dir: Path, names: list[str] | None = None) -> dict[str, np.ndarray]:
    d = voices_dir(meetings_dir)
    if not d.exists():
        return {}
    out = {}
    for f in d.glob("*.npy"):
        name = unquote(f.stem)
        if names is None or name in names:
            v = np.load(f)
            out[name] = v / (np.linalg.norm(v) + 1e-9)
    return out


def save_voiceprint(meetings_dir: Path, name: str, emb: np.ndarray, blend: float = 0.3) -> Path:
    """Store a person's voiceprint, blending into an existing one so it improves over meetings."""
    path = voice_file(meetings_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    if path.exists():
        old = np.load(path)
        emb = (1 - blend) * old / (np.linalg.norm(old) + 1e-9) + blend * emb
        emb = emb / (np.linalg.norm(emb) + 1e-9)
    np.save(path, emb.astype(np.float32))
    return path
