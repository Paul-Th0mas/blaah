import json
import sys
from pathlib import Path

import numpy as np

from meeting_assistant.config import SpeakersCfg
from meeting_assistant.speakers import (
    SpeakerTracker,
    _is_name,
    extract_cues,
    load_namer,
    load_voiceprints,
    proper_nouns,
    save_voiceprint,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from make_fake_meeting import CHAIR, LINES  # noqa: E402

SR = 16000


class FakeEmbedder:
    """Audio filled with the value k means 'speaker k'; each speaker has a fixed voice vector."""

    def __init__(self, noise: float = 0.05, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.base = {}

    def embed(self, audio, sample_rate=SR):
        k = int(round(float(audio[0])))
        if k not in self.base:
            self.base[k] = np.eye(32)[k % 32]
        v = self.base[k] + self.rng.normal(0, self.noise, 32)
        return v / np.linalg.norm(v)


def audio_for(k: int, seconds: float) -> np.ndarray:
    return np.full(int(seconds * SR), float(k), dtype=np.float32)


def tracker(names=("Paul",), **cfg) -> SpeakerTracker:
    return SpeakerTracker(SpeakersCfg(**cfg), list(names), embedder=FakeEmbedder())


# ----- cue extraction -----
def test_cues_cover_address_self_and_thanks():
    assert extract_cues("Dev, start with the cutover.") == [("address", "Dev")]
    assert extract_cues("And Dev, add an idempotency key.") == [("address", "Dev")]
    assert extract_cues("What do you think, Dev?") == [("address", "Dev")]
    assert extract_cues("Karen?") == [("address", "Karen")]
    assert extract_cues("Hi everyone, Raj here, sorry I'm late.") == [("self", "Raj")]
    assert extract_cues("I'm Sarah, the project manager.") == [("self", "Sarah")]
    assert extract_cues("Hi, this is Priya from platform.") == [("self", "Priya")]
    assert extract_cues("Thanks, Karen.") == [("thanks", "Karen")]


def test_cues_ignore_ordinary_words():
    for text in ["Thanks everyone.", "Again, we need it.", "However, no.", "Hiring?", "Noted, sixteen.",
                 "So this is Northwind's issue.", "I'm sorry, I missed that.", "Raj has signed it off.",
                 "Okay, cutover stays.", "Basically, yes."]:
        assert extract_cues(text) == [], text


def test_names_that_are_also_words_need_evidence():
    assert not _is_name("Will") and _is_name("Will", known={"will"})  # configured person
    assert not _is_name("Arjun") and _is_name("Arjun", proper={"arjun"})  # seen mid-sentence
    assert not _is_name("November", proper={"november"})  # stoplist beats mid-sentence capitals
    assert proper_nouns("please tag Arjun on it, and Postgres too") == {"arjun", "postgres"}


# ----- naming a whole meeting -----
def run_meeting(t: SpeakerTracker):
    ids = {name: i + 1 for i, name in enumerate(dict.fromkeys(s for _, s, _ in LINES))}
    out = []
    for _, speaker, text in LINES:
        seconds = max(1.0, len(text.split()) / 2.5)
        out.append((speaker, t.assign(audio_for(ids[speaker], seconds), text)))
    return out


def test_fake_meeting_names_everyone_who_is_named_and_nobody_else():
    t = tracker()
    labelled = run_meeting(t)
    shown = {truth: {t.display(v) for tr, v in labelled if tr == truth} for truth, _ in labelled}
    assert shown["Dev"] == {"Dev"}
    assert shown["Raj"] == {"Raj"}  # self-intro, then confirmed by a hand-off
    assert shown["Karen"] == {"Karen"}
    assert shown["Paul"] == {"Paul"}
    assert shown["Maya"] == {"Maya"}
    # The chair is never named; the wrong hand-off ("Karen, hold the comms" → chair) is outvoted.
    assert len(shown[CHAIR]) == 1 and next(iter(shown[CHAIR])).startswith("Speaker")


def test_names_arrive_after_the_fact_and_relabel_earlier_lines():
    t = tracker()
    changes = []
    t.on_change = lambda names: changes.append(dict(names))
    first = t.assign(audio_for(1, 4), "Sure, the dry run went fine.")
    assert t.display(first).startswith("Speaker")
    t.assign(audio_for(2, 3), "Thanks. Dev, what about the replicas?")
    t.assign(audio_for(1, 4), "They need resizing before we commit.")
    assert t.display(first) == "Dev" and changes[-1] == {first: "Dev"}


def test_short_sentences_never_create_voices():
    t = tracker()
    a = t.assign(audio_for(1, 5), "A long first sentence from one person.")
    b = t.assign(audio_for(2, 0.8), "Yeah.")  # different speaker, but too short to found a voice
    assert len(t.voices) == 1 and b in ("", a)


def test_similar_voices_merge_and_ids_follow():
    t = tracker()
    first = t.assign(audio_for(1, 5), "One sentence.")
    # Force a split: a second voice whose print is then made identical to the first.
    t.voices["S2"] = type(t.voices[first])("S2", t.voices[first].total.copy(), 5.0, 1)
    t._merge_similar()
    assert "S2" not in t.voices and t.resolve("S2") == first


def test_self_cue_from_the_user_and_source_hints():
    t = tracker(names=("Paul", "Paulo"))
    v = t.assign(audio_for(3, 4), "I'm Paulo, joining from the car.")  # alias of the user
    assert t.display(v) == "Paul"
    t2 = tracker()
    w = t2.assign(audio_for(1, 4), "Let me share my screen.", source="me")
    t2.assign(audio_for(1, 4), "Can everyone see it?", source="me")
    assert t2.display(w) == "Paul"  # two mic-only sentences during a call


def test_voiceprints_name_a_voice_on_first_sentence(tmp_path):
    t = tracker()
    t.voiceprints = {"Priya": FakeEmbedder().embed(audio_for(4, 1))}
    v = t.assign(audio_for(4, 4), "Morning all, quick update from platform.")
    assert t.display(v) == "Priya"


def test_voiceprint_files_roundtrip_and_blend(tmp_path):
    a, b = np.eye(4)[0], np.eye(4)[1]
    save_voiceprint(tmp_path, "Dev", a)
    save_voiceprint(tmp_path, "Dev", b, blend=0.5)
    vp = load_voiceprints(tmp_path)["Dev"]
    assert np.isclose(np.linalg.norm(vp), 1.0) and vp[0] > 0 and vp[1] > 0
    assert load_voiceprints(tmp_path, ["Raj"]) == {}


def test_snapshot_and_offline_namer(tmp_path):
    t = tracker()
    run_meeting(t)
    (tmp_path / "speakers.json").write_text(json.dumps(t.snapshot()))
    namer = load_namer(tmp_path)
    named = {namer(v) for v in t.voices}
    assert {"Dev", "Raj", "Karen", "Paul", "Maya"} <= named
    assert namer("Dev") == "Dev"  # plain names pass through (hand-made transcripts)
    assert load_namer(tmp_path / "nowhere")("S1") == "S1"


def test_voiceprints_for_names_with_spaces_or_accents_load_back(tmp_path):
    for name in ("Paul", "José", "Mary Ann"):
        save_voiceprint(tmp_path, name, np.eye(4)[0])
    assert sorted(load_voiceprints(tmp_path)) == ["José", "Mary Ann", "Paul"]
    assert list(load_voiceprints(tmp_path, ["José"])) == ["José"]


def test_name_changes_are_reported_outside_the_lock():
    # on_change re-renders and waits for the UI thread, which may itself be waiting for the
    # tracker (status bar): calling it under the lock would deadlock the two.
    import threading

    t = tracker()
    free = []

    def on_change(_names):
        other = threading.Thread(target=lambda: free.append(t._lock.acquire(timeout=1) and (t._lock.release() or True)))
        other.start()
        other.join()

    t.on_change = on_change
    v = t.assign(audio_for(1, 4), "Hi all, Gavin here.")
    t.set_name(v, "Shaun")
    t.set_attendees(["Paul", "Shaun"])
    assert free and all(free)
    assert t.counts() == (1, 1)


def test_the_ui_and_the_transcriber_can_use_the_tracker_at_once():
    import threading

    t = tracker()
    stop, errors = threading.Event(), []

    def transcriber():
        k = 1
        while not stop.is_set():
            try:
                t.assign(audio_for(k % 40 + 1, 3), f"Thanks, Gavin. Sentence {k} about the migration.")
            except RuntimeError as e:
                errors.append(e)
            k += 1

    def ui():
        while not stop.is_set():
            try:
                t.counts()
                t.voice_summary()
                t.set_attendees(["Paul", "Gavin", "Shaun"])
            except RuntimeError as e:
                errors.append(e)

    threads = [threading.Thread(target=transcriber), threading.Thread(target=ui)]
    for th in threads:
        th.start()
    threading.Event().wait(1.0)
    stop.set()
    for th in threads:
        th.join()
    assert errors == []
