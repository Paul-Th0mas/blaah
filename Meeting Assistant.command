#!/bin/zsh -l
# Starts the meeting assistant in Terminal.
# Opened by double-click, by `open`, or by the "Be my assistant" Siri Shortcut.
printf '\e[8;48;170t'          # resize the window (rows;columns) so the three panels fit
cd "${0:A:h}" || exit 1        # the folder this file lives in
uv run blaah
