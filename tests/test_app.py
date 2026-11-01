import asyncio
import threading

from textual.app import App

from meeting_assistant.app import MeetingApp


def test_ui_updates_asked_for_on_the_ui_thread_still_run():
    # Naming a voice or setting attendees happens in a dialog, on the UI thread; the redraw it
    # triggers must run there and then (call_from_thread refuses the UI thread).
    class Tiny(App):
        _ui = MeetingApp._ui

        def on_mount(self) -> None:
            self._ui_thread = threading.get_ident()

    async def run() -> list[str]:
        app, ran = Tiny(), []
        async with app.run_test():
            app._ui(ran.append, "relabel")
        return ran

    assert asyncio.run(run()) == ["relabel"]
