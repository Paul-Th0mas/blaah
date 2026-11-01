"""Attendee-aware naming, hand-set names, relabelling saved meetings, and the mic/call check.

The sentences mimic real transcriber output from a Google Meet call (names and topics changed):
a colleague's name came out as 'given' / 'Kivinus', another's as an everyday word, and the
user's own voice was never recognised as theirs.
"""

import json

import numpy as np

from meeting_assistant.config import AudioCfg, Settings, SpeakersCfg
from meeting_assistant.meeting import MeetingRecord, attendee_note
from meeting_assistant.naming import load_naming, save_naming
from meeting_assistant.speakers import SpeakerTracker, attendee_match, extract_cues, is_backchannel

PEOPLE = ["Paul", "Shaun", "Gavin", "Leo"]


def test_sound_alike_matching():
    assert attendee_match("given", PEOPLE) == "Gavin"
    assert attendee_match("Kivinus", PEOPLE) == "Gavin"
    assert attendee_match("shown", PEOPLE) == "Shaun"
    assert attendee_match("gavin", PEOPLE) == "Gavin"
    assert attendee_match("Morgan", PEOPLE) is None  # not in the meeting
    assert attendee_match("sorry", PEOPLE) is None  # stoplist
    assert attendee_match("given", []) is None


def test_misheard_names_become_cues_only_in_cue_positions():
    cue = lambda text: extract_cues(text, attendees=PEOPLE)  # noqa: E731
    assert cue("Thank you, given, thank you.") == [("thanks", "Gavin")]
    assert cue("Yeah, I think uh uh Kivinus I forgot to mention did you attend the workshop last week?") == [("address", "Gavin")]
    assert cue("Shown, can you take this one?") == [("address", "Shaun")]
    assert cue("Raise a ticket and they respond pretty quickly, right, Paul sorry no sorry, regarding who.") == [("address", "Paul")]
    # mentions, not addresses
    assert cue("Do we, have they given us something on that?") == []
    assert cue("Given the timeline, we should wait.") == []
    assert cue("As you've shown, the numbers are fine.") == []
    assert cue("If I'm not mistaken, Paul was on the vendor call then.") == []
    assert cue("I thought they left me and Leo, so we're kind of discussing it.") == []


def test_backchannels():
    for text in ["Sorry.", "Cool.", "Sounds good, sounds good, yeah.", "Okay.", "."]:
        assert is_backchannel(text), text
    assert not is_backchannel("Was it the one in Lisbon?")


def _tracker(people=None) -> SpeakerTracker:
    t = SpeakerTracker(SpeakersCfg(), ["Paul"])
    if people:
        t.set_attendees(people)
    return t


def _feed(t: SpeakerTracker, script):
    """Assign sentences with known voices, bypassing audio (as when relabelling a saved meeting)."""
    for vid, text in script:
        if vid not in t.voices:
            from meeting_assistant.speakers import Voice

            t.voices[vid] = Voice(vid, np.zeros(1))
        t.history.append((vid, text, None))
    t._replay()


MEET = [  # shaped like the real call: reactions between a question and its answer
    ("S2", "Raise a ticket and they respond pretty quickly, right, Paul sorry no sorry, regarding who."),
    ("S1", "Vendor support, yeah, they'll do it within half a day."),
    ("S1", "Yeah, I think uh uh Kivinus I forgot to mention did you attend the workshop last week?"),
    ("S4", "Sorry."),
    ("S2", "Was it the one in Lisbon?"),
    ("S1", "Uh I forgot where it was."),
    ("S2", "Oh yeah, that one I went."),
    ("S4", "Cool."),
    ("S4", "Sounds good, sounds good, yeah."),
    ("S1", "Thank you, given, thank you."),
]


def test_meet_call_without_attendees_names_only_the_user():
    t = _tracker()
    _feed(t, MEET)
    assert t.display("S1") == "Paul" and t.display("S2") == "Speaker 2"


def test_attendees_recover_misheard_names_and_skip_reactions():
    t = _tracker()
    _feed(t, MEET)
    t.set_attendees(PEOPLE)  # given later: everything said so far is re-read
    assert t.display("S1") == "Paul"
    assert t.display("S2") == "Gavin"  # the real answer, not the "Sorry." / "Sounds good" in between
    assert t.display("S4") == "Speaker 4"


def test_names_set_by_hand_win_and_move():
    t = _tracker(PEOPLE)
    _feed(t, MEET)
    t.set_name("S2", "Shaun")
    assert t.display("S2") == "Shaun" and "S2" in t.manual
    t.set_name("S4", "Shaun")  # the same name moves to the other voice
    assert t.display("S4") == "Shaun" and "S2" not in t.manual
    assert t.display("S2") == "Gavin"  # back to what the evidence says
    t.set_name("S4", "")
    assert t.display("S4") == "Speaker 4"


def test_relabel_a_saved_meeting(tmp_path):
    rec = MeetingRecord.create(tmp_path)
    with rec.path("transcript.jsonl").open("w") as f:
        for i, (vid, text) in enumerate(MEET):
            f.write(json.dumps({"start": rec.started + i, "end": rec.started + i + 1, "text": text, "speaker": vid}) + "\n")
    rec.path("speakers.json").write_text(json.dumps({v: {"name": "", "seconds": 9.0, "sentences": 3, "evidence": {}} for v in ("S1", "S2", "S4")} | {"_merged": {}}))
    settings = Settings.load(tmp_path / "none.toml")
    tracker, segments, saved = load_naming(rec, settings)
    tracker.set_attendees(PEOPLE)
    tracker.set_name("S4", "Leo")
    save_naming(rec, tracker, segments, saved)
    md = rec.path("transcript.md").read_text()
    assert "Gavin: Was it the one in Lisbon?" in md and "Paul: Thank you, given" in md and "Leo: Sorry." in md
    stored = json.loads(rec.path("speakers.json").read_text())
    assert stored["S4"]["manual"] is True and stored["S2"]["seconds"] == 9.0
    assert MeetingRecord.load(rec.dir).attendees == PEOPLE
    again, _, _ = load_naming(MeetingRecord.load(rec.dir), settings)  # reloads hand-set names
    assert again.display("S4") == "Leo" and again.display("S2") == "Gavin"


def test_attendee_note_marks_the_user():
    assert "Paul (the user), Gavin" in attendee_note(["Paul", "Gavin"], "Paul")
    assert attendee_note([], "Paul") == ""


# ----- mic vs call -----
def _mixed(monkeypatch):
    from meeting_assistant import audio as audio_mod

    monkeypatch.setattr(audio_mod, "resolve_device", lambda _name: None)
    return audio_mod.MixedCapture(AudioCfg(system_audio="all"))


def _levels(m, pairs):
    for mic, system in pairs:
        m._levels.append((mic, system))
        m._levels_total += 1
        m._remote_blocks += system > m.REMOTE_SPEECH_RMS


def test_source_hint_with_meet_background_noise(monkeypatch):
    m = _mixed(monkeypatch)
    _levels(m, [(0.02, 0.06)] * 150)  # 15 s of a remote person talking (echo on the mic)
    _levels(m, [(0.06, 0.007)] * 30)  # 3 s: you talk; the call only carries background noise
    _levels(m, [(0.003, 0.002)] * 20)  # 2 s of quiet
    assert m.source_hint(0.0, 3.0) == "others"
    assert m.source_hint(15.0, 18.0) == "me"  # the old absolute threshold said "others"/nothing here
    assert m.source_hint(18.0, 20.0) is None
    mic, call = m.levels(15.0, 17.9)
    assert abs(mic - 0.06) < 1e-6 and abs(call - 0.007) < 1e-6


def test_source_hint_silent_outside_calls(monkeypatch):
    m = _mixed(monkeypatch)
    _levels(m, [(0.06, 0.0)] * 100)  # in-person meeting: nothing from the call at all
    assert m.source_hint(0.0, 5.0) is None
