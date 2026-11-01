import json
import time

from meeting_assistant.meeting import MeetingRecord, fmt_duration, slugify
from meeting_assistant.summary import split_title


def _write_transcript(folder, texts):
    now = time.time()
    with (folder / "transcript.jsonl").open("w") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({"start": now + i, "end": now + i + 0.5, "text": t}) + "\n")


def test_slugify():
    assert slugify("Weekly Sync: Q3 Roadmap!") == "weekly-sync-q3-roadmap"
    assert slugify("   ") == ""
    assert len(slugify("x" * 100)) == 40


def test_fmt_duration():
    assert fmt_duration(59) == "0m"
    assert fmt_duration(12 * 60) == "12m"
    assert fmt_duration(65 * 60) == "1h 05m"


def test_create_finish_and_rename(tmp_path):
    rec = MeetingRecord.create(tmp_path, stt_model="stt", llm_model="llm")
    assert rec.dir.exists() and rec.id == rec.stamp
    assert json.loads(rec.path("meeting.json").read_text())["llm_model"] == "llm"

    _write_transcript(rec.dir, ["Hello there.", "Second sentence here."])
    rec.finish("Weekly Sync")
    assert rec.id == rec.stamp + "_weekly-sync"
    assert rec.dir.exists() and rec.words == 5 and rec.segments == 2
    assert rec.ended is not None and rec.ended >= rec.started

    loaded = MeetingRecord.load(rec.dir)
    assert loaded.title == "Weekly Sync" and loaded.words == 5


def test_create_with_title_names_folder_immediately(tmp_path):
    rec = MeetingRecord.create(tmp_path, title="1:1 with Sam")
    assert rec.id.endswith("_1-1-with-sam")


def test_list_all_includes_legacy_folders_and_sorts_newest_first(tmp_path):
    legacy = tmp_path / "2026-10-01_090000"
    legacy.mkdir()
    _write_transcript(legacy, ["old meeting words"])
    (tmp_path / "junk").mkdir()  # no transcript → ignored
    newer = MeetingRecord.create(tmp_path, title="New")
    _write_transcript(newer.dir, ["new one"])

    ids = [r.id for r in MeetingRecord.list_all(tmp_path)]
    assert ids[0] == newer.id and ids[1] == "2026-10-01_090000" and len(ids) == 2
    old = MeetingRecord.load(legacy)
    assert old.title == "" and old.words == 3 and old.date_str.startswith("2026-10-01 09:00")


def test_split_title():
    assert split_title('# "Q3 Roadmap"\n**TL;DR:** x') == ("Q3 Roadmap", "**TL;DR:** x")
    assert split_title("**TL;DR:** no title") == ("", "**TL;DR:** no title")


def test_a_meeting_where_nothing_was_said_never_becomes_latest(tmp_path):
    from meeting_assistant.transcript import Segment, TranscriptStore

    done = MeetingRecord.create(tmp_path, title="Standup")
    store = TranscriptStore(done.dir)
    store.add(Segment(time.time(), time.time() + 2, "We ship on Friday."))
    store.close()
    done.finish()
    crashed = MeetingRecord.create(tmp_path)  # the app was killed before anyone spoke
    TranscriptStore(crashed.dir).close()
    assert [r.title for r in MeetingRecord.list_all(tmp_path)] == ["Standup"]


def test_only_old_untouched_empty_meeting_folders_are_pruned(tmp_path):
    from meeting_assistant.transcript import Segment, TranscriptStore

    def folder(hours_ago: float, said: str = "", extra: str = "") -> MeetingRecord:
        rec = MeetingRecord.create(tmp_path, title=f"{hours_ago}h {said} {extra}", started=time.time() - hours_ago * 3600)
        store = TranscriptStore(rec.dir)
        if said:
            store.add(Segment(time.time(), time.time() + 1, said))
        store.close()
        if extra:
            rec.path(extra).write_text("kept\n")
        return rec

    crashed = folder(13)  # killed before anyone spoke, yesterday
    waiting = folder(1)  # another window, waiting for its meeting to start
    talked = folder(13, said="We ship on Friday.")
    briefed = folder(13, extra="briefs.md")
    assert MeetingRecord.prune_empty(tmp_path) == [crashed.dir]
    assert not crashed.dir.exists() and waiting.dir.exists() and talked.dir.exists() and briefed.dir.exists()


def test_a_quiet_meeting_still_open_in_another_window_is_not_pruned(tmp_path):
    from meeting_assistant.transcript import TranscriptStore

    idle = MeetingRecord.create(tmp_path, started=time.time() - 13 * 3600)  # open all day, nobody spoke
    TranscriptStore(idle.dir).close()
    claim = idle.claim()
    assert MeetingRecord.prune_empty(tmp_path) == [] and idle.dir.exists()
    claim.close()  # that window quit (or crashed: the OS drops the lock)
    assert MeetingRecord.prune_empty(tmp_path) == [idle.dir]


def test_meetings_started_in_the_same_second_get_their_own_folders(tmp_path):
    started = time.time()
    a = MeetingRecord.create(tmp_path, started=started)
    b = MeetingRecord.create(tmp_path, started=started)  # a second window, same second
    assert a.dir != b.dir and b.dir.name == a.dir.name + "-2"
