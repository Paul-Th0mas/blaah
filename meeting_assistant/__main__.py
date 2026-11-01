from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

from .audio import FileCapture, list_input_devices
from .config import USER_CONFIG_PATH, ConfigError, Settings, save_user_settings
from .llm import LLMError
from .meeting import MeetingRecord, fmt_duration
from .session import Session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=Path(sys.argv[0]).name or "blaah", description="Blaah — survive the blabber: a local meeting assistant")
    parser.add_argument("command", nargs="?", default="run",
                        choices=["run", "setup", "models", "list", "show", "ask", "enroll", "name", "relabel"],
                        help="run (default) a meeting; setup: your name and model; models: list / use KEY / get KEY; "
                        "list past meetings; show a summary; ask a question; enroll your voice; "
                        "name a voice in a past meeting; relabel a past meeting with its attendees")
    parser.add_argument("target", nargs="*", help="show: meeting id prefix (default latest); ask: the question; "
                        "name: VOICE NAME, e.g. S2 Shaun (no arguments lists the voices)")
    parser.add_argument("--meeting", default="", help="for ask/name/relabel: meeting id prefix (default: latest)")
    parser.add_argument("--people", default="", help='who is in the meeting, e.g. "Shaun, Gavin" (run, relabel)')
    parser.add_argument("--title", default="", help="name for this meeting (can also be set when quitting)")
    parser.add_argument("--config", type=Path, default=None,
                        help=f"use only this config file (default: config.toml + your {USER_CONFIG_PATH})")
    parser.add_argument("--list-devices", action="store_true", help="list audio input devices and exit")
    parser.add_argument("--file", type=Path, help="transcribe an audio file instead of the mic (headless test)")
    parser.add_argument("--fast", action="store_true", help="with --file: feed audio faster than real time")
    parser.add_argument("--headless", action="store_true", help="no TUI; print transcript/notes/briefs to stdout")
    args = parser.parse_args(argv)

    if args.list_devices:
        for idx, name, rate in list_input_devices():
            print(f"{idx:3d}  {name}  ({rate} Hz)")
        return 0

    try:
        settings = Settings.load(args.config)
    except ConfigError as e:
        print(f"Config problem: {e}")
        return 2
    if args.command == "setup":
        return cmd_setup(settings)
    if args.command == "models":
        return cmd_models(settings, args.target)
    if args.command == "list":
        return cmd_list(settings)
    if args.command == "show":
        return cmd_show(settings, args.target[0] if args.target else None)
    if args.command == "ask":
        return cmd_ask(settings, " ".join(args.target), args.meeting)
    if args.command == "enroll":
        return cmd_enroll(settings)
    if args.command == "name":
        return cmd_name(settings, args.target, args.meeting)
    if args.command == "relabel":
        return cmd_relabel(settings, _people(args.people), args.meeting)
    people = _people(args.people)

    source = FileCapture(args.file, settings.audio, realtime=not args.fast) if args.file else None
    if args.file or args.headless:
        return run_headless(settings, source, title=args.title, people=people)

    from .app import MeetingApp

    MeetingApp(
        Session(settings, audio=source, title=args.title, attendees=people),
        new_session=lambda: Session(settings),
        ask_attendees=not people,
    ).run()
    return 0


def _llm(settings: Settings):
    from .providers import make_llm

    return make_llm(settings.llm)


def _print_models(settings: Settings, llm) -> list:
    from .providers import RECOMMENDED

    models = llm.list_models()
    print(f"Provider: {settings.llm.provider}   Model in use: {settings.llm.model}\n")
    if models:
        print("Downloaded models:")
        for i, m in enumerate(models, 1):
            mark = "  ← in use" if m.key == settings.llm.model else ""
            loaded = " (loaded)" if m.loaded else ""
            print(f"  {i}. {m.key:34s} {m.size_gb:5.1f} GB {m.params:>4s}{loaded}{mark}")
    else:
        print("No chat models downloaded yet.")
    have = {m.key for m in models}
    missing = [(k, why) for k, why in RECOMMENDED if k not in have]
    if missing:
        print("\nRecommended (download with `blaah models get KEY`):")
        for key, why in missing:
            print(f"     {key:34s} {why}")
    return models


def cmd_models(settings: Settings, target: list[str]) -> int:
    """`blaah models` lists; `blaah models use KEY` switches; `blaah models get KEY` downloads."""
    llm = _llm(settings)
    try:
        if not target or target[0] == "list":
            _print_models(settings, llm)
            print("\nSwitch with:  blaah models use KEY   (or press m in the app)")
            return 0
        action, key = target[0], " ".join(target[1:]).strip()
        if action not in ("use", "get") or not key:
            print("Usage: blaah models [list] | blaah models use KEY | blaah models get KEY")
            return 1
        if action == "get":
            print(f"Downloading {key} into {settings.llm.provider}…")
            llm.download(key, on_line=lambda line: print("  " + line[-110:]))
        if key not in {m.key for m in llm.list_models()}:
            print(f"{key} isn't downloaded. Download it first:  blaah models get {key}")
            return 1
        path = save_user_settings({"llm": {"model": key}})
        print(f"Blaah will now use {key} (saved in {path}).")
        return 0
    except LLMError as e:
        print(f"Error: {e}")
        return 1


def cmd_setup(settings: Settings) -> int:
    """Interactive first-run setup: your name(s) and the model to use."""
    print("Blaah setup — answers are saved to", USER_CONFIG_PATH, "\n")
    current = ", ".join(settings.user.names)
    answer = input(f"Your name, plus any nicknames people use (comma-separated){f' [{current}]' if current else ''}: ").strip()
    names = [n.strip() for n in answer.split(",") if n.strip()] or settings.user.names
    if not names:
        print("A name is needed so Blaah knows when you're being asked something.")
        return 1
    updates: dict = {"user": {"names": names}}
    try:
        llm = _llm(settings)
        print()
        models = _print_models(settings, llm)
        if models:
            pick = input(f"\nModel number to use [{settings.llm.model}]: ").strip()
            if pick.isdigit() and 1 <= int(pick) <= len(models):
                updates["llm"] = {"model": models[int(pick) - 1].key}
    except LLMError as e:
        print(f"\n(Couldn't reach the model provider: {e}. Your name is saved; pick a model later with `blaah models`.)")
    path = save_user_settings(updates)
    model = updates.get("llm", {}).get("model", settings.llm.model)
    print(f"\nSaved to {path}: name {', '.join(names)}, model {model}. Start with:  uv run blaah")
    return 0


def cmd_list(settings: Settings) -> int:
    records = MeetingRecord.list_all(settings.meetings_dir)
    if not records:
        print(f"No meetings yet in {settings.meetings_dir}")
        return 0
    width = max(len(r.id) for r in records)
    print(f"{'id':{width}s}  {'date':16s} {'length':>7s} {'words':>6s}  title")
    for r in records:
        print(f"{r.id:{width}s}  {r.date_str:16s} {fmt_duration(r.duration):>7s} {r.words:6d}  {r.title}")
    return 0


def find_meeting(settings: Settings, target: str | None) -> MeetingRecord | None:
    records = MeetingRecord.list_all(settings.meetings_dir)
    return next((r for r in records if not target or r.id.startswith(target)), None)


def cmd_ask(settings: Settings, question: str, target: str) -> int:
    if not question.strip():
        print('Usage: blaah ask "what did we decide about X?" [--meeting <id prefix>]')
        return 1
    match = find_meeting(settings, target or None)
    if match is None:
        print(f"No meeting matching {target!r}; try `blaah list`")
        return 1
    from .ask import ask

    llm = _llm(settings)
    try:
        llm.ensure_loaded(lambda m: print(f"  ({m})", flush=True))
        print(f"{match.title or match.id} ({match.date_str})\n")
        print(ask(llm, match, question))
    except LLMError as e:
        print(f"Error: {e}")
        return 1
    return 0


def cmd_show(settings: Settings, target: str | None) -> int:
    match = find_meeting(settings, target)
    if match is None:
        print(f"No meeting matching {target!r}; try `blaah list`")
        return 1
    for name in ("summary.md", "notes.md"):
        path = match.path(name)
        if path.exists():
            print(path.read_text().rstrip())
            break
    else:
        print(f"{match.id}: no summary or notes (transcript only)")
    print(f"\n— files in {match.dir}")
    return 0


ENROLL_TEXT = """\
  "In our planning meeting we went through the roadmap for the next quarter. The database
   migration is on track, but we still need to resize the replicas and agree on a change
   freeze. I'll follow up on the exporter fix and send everyone a short update on Friday.
   If anything changes before then, I'll flag it in the channel straight away." """
ENROLL_SECONDS = 20


def cmd_enroll(settings: Settings) -> int:
    """Record ~20 s of the user's voice and save it as their voiceprint."""
    import numpy as np
    import sounddevice as sd

    from .audio import MicCapture
    from .speakers import VoiceEmbedder, save_voiceprint, voice_file

    if not settings.user.names:
        print("Set your name first (your voiceprint is saved under it):  blaah setup")
        return 1
    name = settings.user.names[0]
    # The same mic your meetings record from (not a Bluetooth headset's, by default), so
    # the voiceprint matches how you sound in them.
    device, why = MicCapture(settings.audio).current_choice()
    print(f"Enrolling the voice of {name!r}." + (f" ({why})" if why else ""))
    print("Loading the voice model…", flush=True)
    embedder = VoiceEmbedder(settings.speakers.model)
    print(f"\nPress Enter, then read this aloud at a normal pace ({ENROLL_SECONDS} s):\n\n{ENROLL_TEXT}\n")
    input()
    sr = settings.audio.sample_rate
    rec = sd.rec(int(ENROLL_SECONDS * sr), samplerate=sr, channels=1, dtype="float32", device=device)
    for left in range(ENROLL_SECONDS, 0, -1):
        print(f"\r  recording… {left:2d} s ", end="", flush=True)
        time.sleep(1)
    sd.wait()
    audio = rec[:, 0]
    if float(np.sqrt(np.mean(audio**2))) < 0.003:
        print("\nThat was nearly silent — check the microphone and try again.")
        return 1
    # Average over 4 s windows: steadier than one long embedding.
    win = 4 * sr
    embs = [embedder.embed(audio[i : i + win], sr) for i in range(0, len(audio) - win + 1, win // 2)]
    print_ = np.mean(embs, axis=0)
    consistency = float(np.mean([e @ (print_ / np.linalg.norm(print_)) for e in embs]))
    voice_file(settings.meetings_dir, name).unlink(missing_ok=True)  # re-enrolling replaces, not blends
    path = save_voiceprint(settings.meetings_dir, name, print_)
    print(f"\nSaved {path} (consistency {consistency:.2f}). Your lines will be labelled {name!r} from now on.")
    return 0


def _people(text: str) -> list[str]:
    return [p.strip() for p in text.split(",") if p.strip()]


def cmd_name(settings: Settings, target: list[str], meeting: str) -> int:
    """List a past meeting's voices, or name one: `name S2 Shaun`."""
    from .naming import describe, load_naming, save_naming

    record = find_meeting(settings, meeting or None)
    if record is None:
        print(f"No meeting matching {meeting!r}; try `blaah list`")
        return 1
    tracker, segments, saved = load_naming(record, settings)
    if not tracker.voices:
        print(f"{record.id}: no voices recorded (speaker identification was off)")
        return 1
    if len(target) >= 2:
        voice, name = target[0].upper(), " ".join(target[1:])
        if tracker.resolve(voice) not in tracker.voices:
            print(f"No voice {voice} in {record.id}. Voices:\n{describe(tracker)}")
            return 1
        tracker.set_name(voice, name)
        if name.lower() not in {a.lower() for a in tracker.attendees}:
            tracker.set_attendees(tracker.attendees + [name])
        save_naming(record, tracker, segments, saved)
        print(f"{voice} is now {name!r}; transcript.md and speakers.json updated.")
    print(f"{record.title or record.id} — voices:\n{describe(tracker)}")
    if len(target) < 2:
        print("\nName one with:  blaah name S2 Shaun" + (f" --meeting {record.id[:17]}" if meeting else ""))
    return 0


def cmd_relabel(settings: Settings, people: list[str], meeting: str) -> int:
    """Re-read a past meeting's name cues with its attendee list (handles misheard names)."""
    from .naming import load_naming, save_naming
    from .session import _with_user

    record = find_meeting(settings, meeting or None)
    if record is None:
        print(f"No meeting matching {meeting!r}; try `blaah list`")
        return 1
    tracker, segments, saved = load_naming(record, settings)
    before = {v: tracker.display(v) for v in tracker.voices}
    if people:
        tracker.set_attendees(_with_user(people, settings.user.names))
    elif not tracker.attendees:
        print('Who was in it? e.g.  blaah relabel --people "Shaun, Gavin"')
        return 1
    save_naming(record, tracker, segments, saved)
    print(f"{record.title or record.id} — attendees: {', '.join(tracker.attendees)}")
    for vid, name, count, _ in tracker.voice_summary():
        change = f"  (was {before[vid]})" if before.get(vid) != name else ""
        print(f"  {vid:4s} {name:16s} {count:4d} sentences{change}")
    print("\nStill unnamed voices can be named by hand: blaah name S3 <name>")
    print("notes.md and summary.md were written before this; they are unchanged.")
    return 0


def _interrupt(*_args) -> None:
    raise KeyboardInterrupt


def run_headless(
    settings: Settings, source: FileCapture | None, title: str = "", people: list[str] | None = None, settle_seconds: float = 2.0
) -> int:
    signal.signal(signal.SIGTERM, _interrupt)  # `kill` wraps up like Ctrl-C instead of dying mid-meeting
    session = Session(settings, audio=source, title=title, attendees=people)
    session.on_segment = lambda seg: print(session.store.line(seg), flush=True)
    session.on_speakers = lambda names: print(f"  (voices: {', '.join(f'{k}={v}' for k, v in sorted(names.items())) or 'none named yet'})", flush=True)
    session.on_notes = lambda notes: print(f"\n=== NOTES ===\n{notes}\n", flush=True)
    session.on_status = lambda comp, msg: print(f"  ({comp}: {msg})", flush=True)

    def run_brief(reason: str) -> None:
        print(f"\n=== BRIEF ({reason}) ===", flush=True)
        try:
            for delta in session.brief(reason):
                print(delta, end="", flush=True)
            stats = session.llm.last_stats
            print(f"\n--- {stats.get('tokens_per_second', 0):.0f} tok/s, ttft {stats.get('time_to_first_token_seconds', 0):.2f}s\n", flush=True)
        except Exception as e:  # LLMError, or whatever a provider plugin raises
            print(f"brief failed: {e}", flush=True)

    session.on_trigger = lambda reason: threading.Thread(target=run_brief, args=(reason,), daemon=True).start()

    print(f"meeting folder: {session.record.dir}", flush=True)
    session.start()
    try:
        if source is not None:
            source.finished.wait()
            print("  (file done)", flush=True)
            time.sleep(settle_seconds)
        else:
            print("  (listening — Ctrl-C to stop)", flush=True)
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print()
    # From here on a second Ctrl-C / kill must not abort the wrap-up half way.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print("  (wrapping up — a few seconds)", flush=True)
    suggested = session.wrap_up()
    record = session.complete(title or suggested)
    if record is None:
        print("Nothing was transcribed — no meeting saved.")
        return 0
    summary = record.path("summary.md")
    if summary.exists():
        print(f"\n=== SUMMARY ===\n{summary.read_text()}")
    print(f"saved: {record.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
