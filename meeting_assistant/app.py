from __future__ import annotations

import re
import threading
import time
from typing import Callable

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Markdown, RichLog, Static
from textual.worker import get_current_worker

from .session import Session
from .transcript import Segment

LEVEL_BARS = " ▁▂▃▄▅▆▇█"
SPEAKER_COLORS = ["cyan", "magenta", "green", "yellow", "bright_blue", "bright_red", "bright_green", "orange1"]


class WrapUpScreen(ModalScreen[str | None]):
    """Shown on quit: runs the wrap-up (final notes + summary), then asks for a title."""

    CSS = """
    WrapUpScreen { align: center middle; }
    #wrap { width: 72; height: auto; border: round $warning; padding: 1 2; background: $surface; }
    #wrap Input { margin-top: 1; }
    """
    BINDINGS = [Binding("escape", "skip", "Save without a title")]

    def __init__(self, session: Session):
        super().__init__()
        self.session = session
        self._ready = False
        self._skip_when_ready = False

    def compose(self) -> ComposeResult:
        with Vertical(id="wrap"):
            yield Static("Wrapping up — final notes and summary…", id="wrap-status")
            yield Input(placeholder="Meeting title  (Enter = save, Esc = no title)", id="title", disabled=True)

    def on_mount(self) -> None:
        self.run_worker(self._wrap, thread=True)

    def _wrap(self) -> None:
        suggested = self.session.wrap_up()
        self.app.call_from_thread(self._done, suggested)

    def _done(self, suggested: str) -> None:
        self._ready = True
        if self._skip_when_ready:
            self.dismiss(None)
            return
        field = self.query_one("#title", Input)
        field.value = self.session.record.title or suggested
        field.disabled = False
        field.focus()
        self.query_one("#wrap-status", Static).update("Saved. Name this meeting:")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def action_skip(self) -> None:
        if self._ready:
            self.dismiss(None)
        else:
            self._skip_when_ready = True
            self.query_one("#wrap-status", Static).update("Finishing up, one moment…")


class AttendeesScreen(ModalScreen[list[str] | None]):
    """Who's in this meeting? Pre-filled from the last meeting; recording runs meanwhile."""

    CSS = """
    AttendeesScreen { align: center middle; }
    #people { width: 76; height: auto; border: round $accent; padding: 1 2; background: $surface; }
    #people Input { margin-top: 1; }
    """
    BINDINGS = [Binding("escape", "skip", "Skip")]

    def __init__(self, prefill: list[str]):
        super().__init__()
        self.prefill = prefill

    def compose(self) -> ComposeResult:
        with Vertical(id="people"):
            yield Static(
                "[b]Who's in this meeting?[/b]  Names separated by commas — they help the app\n"
                "recognise names the transcriber mishears ('given' → Gavin). Enter to confirm, Esc to skip."
            )
            yield Input(value=", ".join(self.prefill), placeholder="e.g. Shaun, Gavin, Priya", id="names")

    def on_mount(self) -> None:
        self.query_one("#names", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss([n.strip() for n in event.value.split(",") if n.strip()])

    def action_skip(self) -> None:
        self.dismiss(None)


class WhoScreen(ModalScreen[None]):
    """Name voices by hand: shows every voice with its latest line; type '2 Shaun'."""

    CSS = """
    WhoScreen { align: center middle; }
    #who { width: 110; height: auto; max-height: 90%; border: round $accent; padding: 1 2; background: $surface; }
    #who Input { margin-top: 1; }
    """
    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(self, session: Session):
        super().__init__()
        self.session = session
        self.rows: list[tuple[str, str, int, str]] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="who"):
            yield Static("", id="voices")
            yield Input(placeholder="number and name, e.g.  2 Shaun   (Enter to apply, Esc to close)", id="assign")

    def on_mount(self) -> None:
        self._refresh()
        self.query_one("#assign", Input).focus()

    def _refresh(self, note: str = "") -> None:
        tracker = self.session.speakers
        self.rows = tracker.voice_summary() if tracker else []
        if not self.rows:
            self.query_one("#voices", Static).update("[b]Who's who[/b]\n\nNo voices yet — wait for a few sentences.")
            return
        lines = ["[b]Who's who[/b]  — name a voice and all its lines update\n"]
        for i, (vid, name, count, latest) in enumerate(self.rows, 1):
            hand = " [dim](set by you)[/dim]" if tracker and vid in tracker.manual else ""
            example = latest if len(latest) <= 70 else latest[:69] + "…"
            lines.append(f" [b]{i}[/b]  {escape(name):14s}{hand}  {count:3d} lines  [dim]“{escape(example)}”[/dim]")
        if note:
            lines.append(f"\n{note}")
        self.query_one("#voices", Static).update("\n".join(lines))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        number, _, name = event.value.strip().partition(" ")
        event.input.value = ""
        if not number.isdigit() or not (1 <= int(number) <= len(self.rows)):
            self._refresh("[red]Type a voice number from the list, then the name — e.g. 2 Shaun[/red]")
            return
        vid = self.rows[int(number) - 1][0]
        self.session.set_speaker_name(vid, name)
        self._refresh(f"[green]{number} is now {escape(name.strip())}[/green]" if name.strip() else f"[green]{number}: name cleared[/green]")

    def action_close(self) -> None:
        self.dismiss(None)


class ModelScreen(ModalScreen[str | None]):
    """Pick the model Blaah uses (from the provider's downloaded models); returns its key."""

    CSS = """
    ModelScreen { align: center middle; }
    #models { width: 90; height: auto; max-height: 90%; border: round $accent; padding: 1 2; background: $surface; }
    #models Input { margin-top: 1; }
    """
    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(self, session: Session):
        super().__init__()
        self.session = session
        self.models: list = []

    def compose(self) -> ComposeResult:
        with Vertical(id="models"):
            yield Static("Looking for models…", id="list")
            yield Input(placeholder="number of the model to use (Enter), Esc to close", id="pick")

    def on_mount(self) -> None:
        self.query_one("#pick", Input).focus()
        self.run_worker(self._load, thread=True)

    def _load(self) -> None:
        try:
            models, error = self.session.llm.list_models(), ""
        except Exception as e:  # provider down, LM Studio missing…
            models, error = [], str(e)
        self.app.call_from_thread(self._show, models, error)

    def _show(self, models: list, error: str) -> None:
        self.models = models
        current = self.session.llm.cfg.model
        lines = [f"[b]Model[/b]  — {escape(self.session.settings.llm.provider)}, in use: [b]{escape(current)}[/b]\n"]
        if error:
            lines.append(f"[red]{escape(error)}[/red]")
        for i, m in enumerate(models, 1):
            mark = "  [green]← in use[/green]" if m.key == current else ""
            lines.append(f" [b]{i}[/b]  {escape(m.key):34s} {m.size_gb:4.1f} GB{mark}")
        if not models and not error:
            lines.append("No chat models downloaded. In a terminal:  blaah models get qwen/qwen3.5-4b")
        lines.append("\n[dim]Your choice is remembered for next time. Download more with `blaah models get KEY`.[/dim]")
        self.query_one("#list", Static).update("\n".join(lines))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        pick = event.value.strip()
        if pick.isdigit() and 1 <= int(pick) <= len(self.models):
            self.dismiss(self.models[int(pick) - 1].key)
        else:
            event.input.value = ""

    def action_close(self) -> None:
        self.dismiss(None)


class NameScreen(ModalScreen[list[str] | None]):
    """First run: who are you? (the name that triggers a brief)."""

    CSS = """
    NameScreen { align: center middle; }
    #me { width: 76; height: auto; border: round $warning; padding: 1 2; background: $surface; }
    #me Input { margin-top: 1; }
    """
    BINDINGS = [Binding("escape", "skip", "Skip")]

    def compose(self) -> ComposeResult:
        with Vertical(id="me"):
            yield Static(
                "[b]Welcome to Blaah.[/b]  What's your name? Add nicknames people use, separated by commas.\n"
                "When someone says it, Blaah hands you a reply. Saved for next time."
            )
            yield Input(placeholder="e.g. Priya, Pri", id="names")

    def on_mount(self) -> None:
        self.query_one("#names", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        names = [n.strip() for n in event.value.split(",") if n.strip()]
        if names:
            self.dismiss(names)

    def action_skip(self) -> None:
        self.dismiss(None)


class NewMeetingScreen(ModalScreen[str | None]):
    """Asks what to do with the current meeting before starting a fresh one."""

    CSS = """
    NewMeetingScreen { align: center middle; }
    #choice { width: 64; height: auto; border: round $warning; padding: 1 2; background: $surface; }
    """
    BINDINGS = [
        Binding("s", "pick('save')", "Save it"),
        Binding("d", "pick('delete')", "Delete it"),
        Binding("escape", "pick(None)", "Cancel"),
    ]

    def __init__(self, sentences: int):
        super().__init__()
        self.sentences = sentences

    def compose(self) -> ComposeResult:
        with Vertical(id="choice"):
            yield Static(
                f"[b]Start a new meeting[/b]\n\nThis one has {self.sentences} transcribed sentences.\n\n"
                "  [b]s[/b]  save it first — final notes, summary, title\n"
                "  [b]d[/b]  delete it — transcript, notes and briefs are removed\n"
                "  [b]Esc[/b] keep going with this meeting"
            )

    def action_pick(self, choice: str | None) -> None:
        self.dismiss(choice)


class MeetingApp(App):
    TITLE = "Blaah"
    CSS = """
    #body { height: 1fr; }
    #left { width: 55%; border: round $primary; padding: 0 1; }
    #right { width: 45%; }
    .title { text-style: bold; color: $accent; }
    #transcript { height: 1fr; background: transparent; }
    #live { height: auto; min-height: 2; text-style: italic; }
    #notes-box { height: 50%; border: round $primary; padding: 0 1; }
    #brief-box { height: 50%; border: round $warning; padding: 0 1; }
    #status { height: 1; background: $surface; padding: 0 1; }
    """
    BINDINGS = [
        Binding("b", "brief", "Brief me"),
        Binding("n", "notes", "Refresh notes"),
        Binding("p", "pause", "Pause mic"),
        Binding("w", "who", "Who's who"),
        Binding("a", "attendees", "Attendees"),
        Binding("m", "model", "Model"),
        Binding("r", "new_meeting", "New meeting"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, session: Session, new_session: Callable[[], Session] | None = None, ask_attendees: bool = True):
        super().__init__()
        self.session = session
        self.new_session = new_session or (lambda: Session(session.settings))
        self.ask_attendees = ask_attendees
        self._status = {"stt": "starting", "llm": "starting", "notes": "waiting for speech"}
        self._brief_text = ""
        self._brief_dirty = False
        self._brief_started = 0.0
        self._busy = False  # quitting or switching meetings
        self._ui_thread: int | None = None
        self.sub_title = session.record.title or session.record.date_str

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield Static("TRANSCRIPT", classes="title")
                yield RichLog(id="transcript", wrap=True, markup=True, auto_scroll=True)
                yield Static("", id="live")
            with Vertical(id="right"):
                with VerticalScroll(id="notes-box"):
                    yield Static("RUNNING NOTES", classes="title")
                    yield Markdown("_waiting for enough speech…_", id="notes")
                with VerticalScroll(id="brief-box"):
                    yield Static("BRIEF  — press b, or say my name", classes="title", id="brief-title")
                    yield Markdown("", id="brief")
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self._ui_thread = threading.get_ident()
        self._attach(self.session)
        self.set_interval(0.5, self._refresh_status)
        self.set_interval(0.15, self._flush_brief)

    def _attach(self, s: Session) -> None:
        """Point the UI at a session and start it in the background."""
        self.session = s
        s.on_segment = lambda seg: self._ui(self._add_segment, seg)
        s.on_live = lambda text: self._ui(self._set_live, text)
        s.on_notes = lambda notes: self._ui(self._set_notes, notes)
        s.on_status = lambda comp, msg: self._ui(self._set_status, comp, msg)
        s.on_trigger = lambda reason: self._ui(self._triggered, reason)
        s.on_speakers = lambda _names: self._ui(self._relabel)
        self.sub_title = s.record.title or s.record.date_str
        self.run_worker(self._start_session, thread=True, name="start")  # recording starts right away
        if self.ask_attendees and not s.attendees[:-1]:  # nobody but the user yet
            self.action_attendees()
        if not s.settings.user.names:  # first run: shown on top of the attendee box
            self.push_screen(NameScreen(), callback=self._set_my_name)

    # ----- background threads → UI thread -----
    def _ui(self, fn, *args) -> None:
        if threading.get_ident() == self._ui_thread:
            fn(*args)  # already on the UI thread, e.g. a name set in a dialog
            return
        try:
            self.call_from_thread(fn, *args)
        except Exception:
            pass  # app is shutting down

    def _start_session(self) -> None:
        try:
            self.session.start()
        except Exception as e:  # e.g. no microphone
            self._ui(self._set_status, "stt", f"error: {e}")

    # ----- UI updates -----
    def _line(self, seg: Segment) -> str:
        who = self.session.store.who(seg)
        if not who:
            return f"[dim]{seg.clock}[/] {escape(seg.text)}"
        tracker = self.session.speakers
        vid = tracker.resolve(seg.speaker) if tracker else seg.speaker
        color = SPEAKER_COLORS[int(vid[1:]) % len(SPEAKER_COLORS)] if vid[1:].isdigit() else "white"
        if tracker and who == tracker.user:
            who += " (you)"
        return f"[dim]{seg.clock}[/] [bold {color}]{escape(who)}:[/] {escape(seg.text)}"

    def _add_segment(self, seg: Segment) -> None:
        self.query_one("#transcript", RichLog).write(self._line(seg))

    def _relabel(self) -> None:
        """A voice got (or changed) its name: redraw the transcript with current names."""
        log = self.query_one("#transcript", RichLog)
        log.clear()
        for seg in self.session.store.segments():
            log.write(self._line(seg))

    def _set_live(self, text: str) -> None:
        self.query_one("#live", Static).update(f"[dim]live ▸[/] {escape(text[-400:])}" if text else "")

    def _set_notes(self, notes: str):
        return self.query_one("#notes", Markdown).update(notes)

    def _set_status(self, component: str, message: str) -> None:
        self._status[component] = message
        if component == "audio":  # device switches deserve a visible note
            self.notify(message, title="Audio")

    def _triggered(self, reason: str) -> None:
        self.bell()
        self.notify(reason, title="Blaah", severity="warning")
        if self.session.settings.triggers.auto_brief:
            self.action_brief(reason)

    def _refresh_status(self) -> None:
        s = self.session
        level = min(len(LEVEL_BARS) - 1, int(s.audio.level * 60))
        mic = f"{'PAUSED' if s.audio.paused else 'mic'} {LEVEL_BARS[level]}"
        if not s.transcriber.ready.is_set():
            stt = f"STT {self._status['stt']}"
        else:
            behind = s.transcriber.behind_seconds
            stt = "STT live" if behind < 2 else f"STT {behind:.0f}s behind, catching up"
            if s.audio.dropped:
                stt += f" ⚠ {s.audio.dropped} s of audio dropped"
        llm = f"{s.llm.cfg.model} {self._status['llm']}"
        names = ", ".join(s.settings.user.names)
        system = getattr(s.audio, "system", None)
        extra = f" │ {system.status}" if system is not None else ""
        if s.speakers is not None:
            count, named = s.speakers.counts()
            voices = f"{count} voices, {named} named" if count else self._status.get("voices", "")
            extra += f" │ {voices}"
        self.query_one("#status", Static).update(
            escape(f"{mic} ({s.audio.device_name}){extra} │ {stt} │ {llm} │ notes: {self._status['notes']} │ listening for: {names}")
        )

    async def _flush_brief(self) -> None:
        if self._brief_dirty:
            self._brief_dirty = False
            # The model writes one line per item; Markdown would merge single newlines.
            text = re.sub(r"(?<!\n)\n(?!\n)", "\n\n", self._brief_text)
            await self.query_one("#brief", Markdown).update(text)

    # ----- actions -----
    def action_brief(self, reason: str = "requested") -> None:
        self.run_worker(lambda: self._brief_worker(reason), thread=True, group="brief", exclusive=True)

    def _brief_worker(self, reason: str) -> None:
        worker = get_current_worker()
        self._brief_started = time.perf_counter()
        self._ui(self.query_one("#brief-title", Static).update, f"BRIEF  — {reason} — thinking…")
        self._brief_text, self._brief_dirty = "", True
        text = ""
        stream = self.session.brief(reason)
        try:
            for delta in stream:
                if worker.is_cancelled:
                    return  # a newer brief took over
                text += delta
                self._brief_text, self._brief_dirty = text, True
        except Exception as e:
            self._brief_text, self._brief_dirty = f"**Brief failed:** {escape(str(e))}", True
            return
        finally:
            stream.close()  # stops generation now, and lets notes run again
        took = time.perf_counter() - self._brief_started
        tps = self.session.llm.last_stats.get("tokens_per_second", 0.0)
        self._ui(
            self.query_one("#brief-title", Static).update,
            f"BRIEF  — {reason} — {took:.1f}s, {tps:.0f} tok/s",
        )

    def action_notes(self) -> None:
        self.session.tracker.refresh_now()
        self._status["notes"] = "refresh requested"

    def action_pause(self) -> None:
        self.session.audio.paused = not self.session.audio.paused

    def action_quit(self) -> None:
        if self._busy:
            return
        self._busy = True
        self.workers.cancel_group(self, "brief")
        self.push_screen(WrapUpScreen(self.session), callback=self._finish)

    def _finish(self, title: str | None) -> None:
        record = self.session.complete(title)
        self.exit(message=f"Saved: {record.dir}" if record else "Nothing was transcribed — no meeting saved.")

    # ----- people -----
    def action_attendees(self) -> None:
        current = [a for a in self.session.attendees if a.lower() not in {u.lower() for u in self.session.settings.user.names}]
        self.push_screen(AttendeesScreen(current or self._last_attendees()), callback=self._set_attendees)

    def _last_attendees(self) -> list[str]:
        """Everyone from the most recent earlier meeting that had a list, minus the user."""
        from .meeting import MeetingRecord

        users = {u.lower() for u in self.session.settings.user.names}
        for rec in MeetingRecord.list_all(self.session.settings.meetings_dir):
            if rec.dir != self.session.record.dir and rec.attendees:
                return [a for a in rec.attendees if a.lower() not in users]
        return list(self.session.settings.speakers.people)

    def _set_attendees(self, names: list[str] | None) -> None:
        if names is None:
            return
        self.session.set_attendees(names)
        self.notify(", ".join(self.session.attendees) or "nobody", title="Attendees")

    def _set_my_name(self, names: list[str] | None) -> None:
        if not names:
            self.notify("No name set — Blaah can't tell when you're asked something. Run `blaah setup` later.",
                        severity="warning")
            return
        from .config import save_user_settings

        save_user_settings({"user": {"names": names}})
        self.session.set_user_names(names)
        self.notify(f"Listening for {', '.join(names)}", title="Saved")

    def action_model(self) -> None:
        if self._busy:
            return
        self.push_screen(ModelScreen(self.session), callback=self._use_model)

    def _use_model(self, key: str | None) -> None:
        if not key or key == self.session.llm.cfg.model:
            return
        self._status["llm"] = f"switching to {key}…"
        self.run_worker(lambda: self._switch_model(key), thread=True, group="model", exclusive=True)

    def _switch_model(self, key: str) -> None:
        if not self.session.switch_model(key):
            return  # the session says why, and is back on the previous model
        from .config import save_user_settings

        save_user_settings({"llm": {"model": key}})  # remembered only once it has worked
        self._ui(self.notify, f"Using {key} from now on.", title="Model")

    def action_who(self) -> None:
        if self.session.speakers is None:
            self.notify("Speaker identification is off ([speakers] enabled = false).")
            return
        self.push_screen(WhoScreen(self.session))

    # ----- new meeting: save or delete this one, then start fresh -----
    def action_new_meeting(self) -> None:
        if self._busy:
            return
        self.push_screen(NewMeetingScreen(len(self.session.store)), callback=self._new_meeting_choice)

    def _new_meeting_choice(self, choice: str | None) -> None:
        if choice is None:
            return
        self._busy = True
        self.workers.cancel_group(self, "brief")
        if choice == "save":
            self.push_screen(WrapUpScreen(self.session), callback=self._saved_then_restart)
        else:
            self._status["stt"] = "deleting this meeting…"
            self.run_worker(self._delete_then_restart, thread=True, group="switch")

    def _saved_then_restart(self, title: str | None) -> None:
        record = self.session.complete(title)
        self.notify(f"Saved {record.id}" if record else "Nothing was transcribed — nothing saved.", title="Meeting saved")
        self._restart()

    def _delete_then_restart(self) -> None:
        self.session.discard()
        self._ui(self.notify, "Previous meeting deleted.", title="New meeting")
        self._ui(self._restart)

    def _restart(self) -> None:
        old = self.session
        quiet = lambda *_args: None  # noqa: E731 — late events from the old meeting are ignored
        old.on_segment = old.on_live = old.on_notes = old.on_status = old.on_trigger = old.on_speakers = quiet
        self._reset_view()
        self._attach(self.new_session())
        self._busy = False

    def _reset_view(self) -> None:
        self.query_one("#transcript", RichLog).clear()
        self.query_one("#live", Static).update("")
        self.query_one("#notes", Markdown).update("_waiting for enough speech…_")
        self.query_one("#brief", Markdown).update("")
        self.query_one("#brief-title", Static).update("BRIEF  — press b, or say my name")
        self._brief_text, self._brief_dirty = "", False
        self._status = {"stt": "starting", "llm": "starting", "notes": "waiting for speech"}
