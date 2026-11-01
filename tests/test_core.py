import time
from types import SimpleNamespace

from meeting_assistant.config import LlmCfg, Settings, TrackerCfg
from meeting_assistant.llm import LLM
from meeting_assistant.tracker import Tracker
from meeting_assistant.transcriber import Timed, _split_sentences, _text
from meeting_assistant.transcript import Segment, TranscriptStore
from meeting_assistant.triggers import NameDetector


# ----- name detection -----
def test_detector_exact_and_case_insensitive():
    d = NameDetector(["Paul"], cooldown_seconds=0)
    assert d.check("okay so paul can you") == "paul"
    assert d.check("nothing here") is None


def test_detector_fuzzy_catches_transcription_slips_but_not_common_words():
    d = NameDetector(["Paul"], cooldown_seconds=0)
    assert d.check("it is Pauls turn") == "Pauls"
    assert d.check("pull the data please") is None
    assert d.check("the pool is cool") is None


def test_detector_short_names_are_exact_only():
    d = NameDetector(["Al"], cooldown_seconds=0)
    assert d.check("Al, thoughts?") == "Al"
    assert d.check("all good") is None  # no fuzzy for 2-letter names


def test_detector_cooldown_suppresses_repeats():
    d = NameDetector(["Paul"], cooldown_seconds=60)
    assert d.check("Paul?") == "Paul"
    assert d.check("Paul again") is None
    d.last_fired = time.time() - 61
    assert d.check("Paul again") == "Paul"


# ----- sentence splitting -----
def _dated(*pieces: str) -> list[Timed]:
    return [Timed(SimpleNamespace(text=p), t=float(i)) for i, p in enumerate(pieces)]


def test_split_sentences_groups_on_terminal_punctuation():
    items = _dated(" Okay", ",", " let", "'s", " go", ".", " Next", " item", "?", " Un", "finished")
    groups = _split_sentences(items)
    assert [_text(g) for g in groups] == ["Okay, let's go.", "Next item?", "Unfinished"]
    assert groups[0][0].t == 0.0 and groups[1][-1].t == 8.0


def test_split_sentences_empty():
    assert _split_sentences([]) == []


# ----- transcript store -----
def test_store_since_index_and_recent_window(tmp_path):
    store = TranscriptStore(tmp_path)
    now = time.time()
    store.add(Segment(now - 500, now - 499, "old news."))
    store.add(Segment(now - 10, now - 9, "fresh news."))
    store.set_live("and more coming")

    new, cursor = store.since_index(0)
    assert [s.text for s in new] == ["old news.", "fresh news."] and cursor == 2
    assert store.since_index(cursor) == ([], 2)

    recent = store.recent(60)
    assert "fresh news." in recent and "old news." not in recent
    assert recent.endswith("[live] and more coming")
    assert "[live]" not in store.recent(60, include_live=False)

    store.close()
    assert (tmp_path / "transcript.jsonl").read_text().count("\n") == 2


# ----- tracker thread lifecycle -----
def test_tracker_can_be_stopped_and_joined():
    # Regression: a `_stop` attribute on a Thread subclass shadows Thread._stop() and breaks join().
    tracker = Tracker(TrackerCfg(interval_seconds=60), llm=None, store=TranscriptStore(), user_names=["Paul"], max_tokens=10)
    tracker.start()
    tracker.stop()
    tracker.join(timeout=5)
    assert not tracker.is_alive()


# ----- config -----
def test_settings_overlay_ignores_unknown_keys(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[user]\nnames = ["Sam", "Sammy"]\nbogus = 1\n[llm]\nmodel = "tiny"\n')
    s = Settings.load(cfg)
    assert s.user.names == ["Sam", "Sammy"]
    assert s.llm.model == "tiny"
    assert s.llm.context_length == 4096  # default kept
    assert s.audio.sample_rate == 16000


# ----- LM Studio client parsing -----
def test_llm_result_parsing_handles_string_and_parts():
    llm = LLM(LlmCfg())
    r = llm._result({"output": [{"type": "message", "content": "hi"}], "stats": {"tokens_per_second": 12.5}})
    assert r.text == "hi" and r.tokens_per_second == 12.5
    r = llm._result({"output": [{"type": "message", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]})
    assert r.text == "ab"


def test_llm_reload_decision():
    llm = LLM(LlmCfg(context_length=4096, parallel=1))
    assert llm._needs_reload({"config": {"context_length": 8192, "parallel": 4}})  # too many slots
    assert llm._needs_reload({"config": {"context_length": 4096, "parallel": 2}})
    assert llm._needs_reload({"config": {"context_length": 2048, "parallel": 1}})  # too small
    assert not llm._needs_reload({"config": {"context_length": 4096, "parallel": 1}})
    assert not llm._needs_reload({"config": {"context_length": 61952, "parallel": 1}})  # MLX reports big


def test_llm_body_turns_reasoning_off_only_when_supported():
    llm = LLM(LlmCfg(model="m"))
    assert llm._body("sys", "hi", 10, None, stream=False)["reasoning"] == "off"
    llm.reasoning_off = False
    assert "reasoning" not in llm._body("", "hi", 10, 0.5, stream=True)


# ----- review fixes: detector counts, notes backlog and failures, config checks, LM Studio errors -----
def test_detector_counts_mentions_without_a_cooldown():
    d = NameDetector(["Paul"], cooldown_seconds=60)
    assert d.count("Paul, and then Pauls idea, thanks Paul") == 3
    assert d.check("Paul?") == "Paul" and d.count("Paul?") == 1  # counting ignores the cooldown
    assert NameDetector([], 60).count("Paul") == 0


def _store_of(segments: int, words: int) -> TranscriptStore:
    store, t = TranscriptStore(), time.time() - 600
    for i in range(segments):
        store.add(Segment(t + i * 5, t + i * 5 + 4, " ".join(f"w{i}_{j}" for j in range(words))))
    return store


class _EchoLLM:
    last_stats: dict = {}

    def __init__(self, fail_with: Exception | None = None):
        self.prompts, self.fail_with = [], fail_with

    def stream(self, system, prompt, *, max_tokens=512, temperature=None):
        self.prompts.append(prompt)
        yield "## Now discussing\n"
        if self.fail_with:
            raise self.fail_with
        yield "the cutover"


def test_a_notes_backlog_is_folded_in_pieces_without_skipping_any():
    llm, store = _EchoLLM(), _store_of(15, 100)  # 1500 words: notes failed or waited ~10 minutes
    tracker = Tracker(TrackerCfg(), llm, store, ["Paul"], max_tokens=100)
    assert tracker.update(force=True) is True  # more to come
    assert "w0_0" in llm.prompts[0] and "w11_99" in llm.prompts[0] and "w12_0" not in llm.prompts[0]
    assert tracker._cursor == 12
    assert tracker.update(force=True) is False
    assert "w12_0" in llm.prompts[1] and tracker._cursor == 15


def test_any_provider_error_keeps_the_notes_and_the_thread():
    for error in (ConnectionResetError(54, "Connection reset by peer"), RuntimeError("plugin bug")):
        tracker = Tracker(TrackerCfg(min_new_words=1), _EchoLLM(fail_with=error), _store_of(3, 10), ["Paul"], max_tokens=100)
        tracker.notes = "## Now discussing\nCutover moved to Friday"
        statuses = []
        tracker.on_status = statuses.append
        assert tracker.update(force=True) is False
        assert tracker.notes == "## Now discussing\nCutover moved to Friday" and tracker._cursor == 0  # retried later
        assert statuses[-1].startswith("notes failed")


def test_config_values_are_checked_and_a_lone_name_is_a_list(tmp_path):
    import pytest

    from meeting_assistant.config import ConfigError

    cfg = tmp_path / "config.toml"
    cfg.write_text('[user]\nnames = "Paul"\n[tracker]\ninterval_seconds = 30\n')
    s = Settings.load(cfg)
    assert s.user.names == ["Paul"] and NameDetector(s.user.names, 0).check("I have a question") is None
    assert s.tracker.interval_seconds == 30.0
    for bad, message in (('[tracker]\ninterval_seconds = "45s"', "should be a number"),
                         ("[audio]\nprefer_builtin_mic = 1", "true or false"),
                         ("[llm]\ncontext_length = 4096.5", "whole number"),
                         ("[user\nnames = []", "config.toml")):
        cfg.write_text(bad + "\n")
        with pytest.raises(ConfigError, match=message):
            Settings.load(cfg)


def test_provider_specific_llm_settings_reach_the_provider(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[llm]\nprovider = "ollama"\nkeep_alive = "10m"\nmodel = "llama3"\n')
    s = Settings.load(cfg)
    assert s.llm.extra == {"keep_alive": "10m"} and s.llm.model == "llama3"


def _cut_off_server(reset: bool) -> int:
    import socket
    import struct
    import threading

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def serve():
        conn, _ = srv.accept()
        conn.recv(65536)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 5000\r\n\r\n"
                     b'data: {"type":"message.delta","content":"## Now discus"}\n\n')
        if reset:
            time.sleep(0.1)
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        conn.close()  # LM Studio quit or crashed mid-reply
        srv.close()

    threading.Thread(target=serve, daemon=True).start()
    return srv.getsockname()[1]


def test_a_reply_cut_off_mid_stream_is_an_error_not_a_short_reply():
    import pytest

    from meeting_assistant.llm import LLMError

    for reset in (False, True):
        llm = LLM(LlmCfg(base_url=f"http://127.0.0.1:{_cut_off_server(reset)}"))
        with pytest.raises(LLMError):
            list(llm.stream("", "notes please"))


def test_a_missing_lms_command_is_an_llm_error(monkeypatch):
    import pytest

    from meeting_assistant.llm import LLMError

    monkeypatch.setattr(LLM, "server_up", lambda _self: True)  # LM Studio app open, CLI never bootstrapped
    monkeypatch.setattr(LLM, "models", lambda _self: [{"key": "qwen/qwen3.5-4b", "type": "llm", "loaded_instances": []}])
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(LLMError, match="lms bootstrap"):
        LLM(LlmCfg()).ensure_loaded()


def test_model_download_progress_is_shown_a_few_times_not_every_tick(tmp_path, monkeypatch):
    lms = tmp_path / "lms"
    lms.write_text("#!/bin/sh\nfor i in $(seq 1 300); do printf '\\rDownloading  %s%%' $i; done; printf '\\nDone\\n'\n")
    lms.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    lines = []
    LLM(LlmCfg()).download("qwen/qwen3.5-9b", on_line=lines.append)
    assert lines[0] == "Downloading  1%" and lines[-1] == "Done" and len(lines) <= 3


def test_draft_skipping_only_patches_models_whose_decode_it_can_mimic():
    from parakeet_mlx.parakeet import ParakeetCTC, ParakeetTDT

    from meeting_assistant.transcriber import _skippable_drafts

    class TDT(ParakeetTDT):
        def __init__(self):
            self.decode = lambda *_a, **_k: "decoded"

    class CTC(ParakeetCTC):
        def __init__(self):
            self.decode = lambda *_a, **_k: "decoded"

    tdt = TDT()
    want, calls = _skippable_drafts(tdt)
    want[0] = False
    assert tdt.decode() == "decoded" and tdt.decode() == ([[]], [None])  # 2nd call = the draft
    ctc = CTC()
    real = ctc.decode
    _skippable_drafts(ctc)
    assert ctc.decode is real  # CTC returns tokens alone; a tuple stub would break the live text


def test_introductions_are_not_someone_asking_you():
    d = NameDetector(["Paul"], cooldown_seconds=0)
    for intro in ("Hi all, Paul here.", "This is Paul from platform.", "I'm Paul.", "Paul speaking.",
                  "Morning, my name is Paul and I run the data team."):
        assert d.check(intro) is None and d.count(intro) == 0, intro
    for ask in ("Paul, can you take this?", "what do you think, Paul?", "Can Paul here confirm the date?",
                "I'm Paul's backup this week, so Paul can you check?"):
        assert d.check(ask) == "Paul", ask
