# Blaah

*Survive the blabber.*

Long meetings are mostly blah blah blah — and somewhere in the middle of it your attention
drifts. Then suddenly your name is called, everyone's looking at you, and you have no idea
what was just asked.

**Blaah is there to rescue you.** It quietly listens along, and the moment your name is
called it gathers what's been going on and hands you a suitable reply, so you're back in
tune with the blabber in seconds. The rest of the time it's a plain meeting assistant:
running notes, who said what, a summary at the end, and answers about past meetings.

Nothing leaves the laptop: speech-to-text runs on the GPU via MLX, and the language model
runs in LM Studio.

Nothing leaves the laptop: speech-to-text runs on the GPU via MLX, and the language model
runs in LM Studio.

## How it works

```
mic ──► Parakeet (MLX, in-process) ──► transcript ──┬──► name detector ──► notification + brief
         live text every ~2 s                        ├──► tracker (every 45 s) ──► running notes
                                                     └──► brief (on demand) ──► "right now / last said / you could say"
```

Two models, two jobs:

| Role | Model | Where it runs | Why |
|---|---|---|---|
| Speech-to-text | `parakeet-tdt-0.6b-v3` (0.6B) | in-process, MLX | Fastest STT on Apple Silicon (~2× faster than Whisper), 25 languages, ~1.2 GB |
| Notes + briefs | `qwen/qwen3.5-4b` (configurable) | LM Studio | Best speed/quality on 16 GB — see *Performance* below. Any LM Studio model works; swap in `config.toml` |

The app talks to LM Studio's native REST API (`/api/v1/chat`) rather than the OpenAI
endpoint because that is the only way to switch a thinking model's reasoning **off** per
request. With thinking on, Qwen spends seconds reasoning before the first word of a brief.

Parakeet finalises words ~20 s after they are spoken (its right-context window), so name
detection and briefs also use the live "draft" text, which arrives within ~2 s.

**Keeping up under load.** Every model call re-reads the 20 s look-ahead window (~0.5 s of GPU
time on an M5) and re-decodes a live draft of it (~0.4 s), however little new audio it gets.
During a video call, with Qwen writing notes, that can exceed real time — an early Google Meet
test drifted 5 minutes behind. Now: audio goes in 2 s chunks; anything that piles up is
processed in one batch with the draft skipped (identical final text, ~40 % cheaper); and notes
refreshes wait while transcription is behind. In a 3-minute real-time test with an LLM
generating non-stop alongside, transcription stayed within 4 s of live (0.8 s on average).
The status bar shows `STT live` or how far behind it is.

## Performance on a 16 GB Mac (measured, M5)

The 9B model needs ~7 GB of GPU memory; with Parakeet (~1.5 GB), macOS and a browser the
machine swaps, and the LLM slows down a lot. How LM Studio loads the model matters:

| LM Studio load config | Generation | Prompt processing (1.2k tokens) | A ~110-word brief |
|---|---|---|---|
| 8192 context, 4 parallel slots (LM Studio's default here) | ~8 tok/s | ~8 s | ~25 s |
| **4096 context, 1 slot (what the app loads)** | ~15 tok/s | ~4 s | ~10 s |

The app loads the model with the lighter config and reloads it if LM Studio already has it
loaded heavier. Notes refreshes stream and **yield to a brief** mid-generation, so a brief never
waits behind a notes update even with one slot.

Model comparison on the same recorded meeting (each model loaded alone, Parakeet resident),
via `uv run python scripts/compare_models.py meetings/<run>/transcript.jsonl <model> ...`:

| Model | Size | Notes refresh | Brief (first word → done) | Verdict |
|---|---|---|---|---|
| Qwen3.5-9B (GGUF Q4) | 6.5 GB | 8.4 s | 2.0 s → 7.9 s | Best judgement, but slow and memory-hungry |
| **Qwen3.5-4B (MLX 4-bit)** | 3.1 GB | 4.0 s | **0.8 s → 4.1 s** | **Default.** Close to the 9B in quality at half the latency |
| Granite 4.0 H-Micro (MLX) | 1.8 GB | 2.7 s | 0.8 s → 2.6 s | Fastest, but ignores the notes format and invents answers — unsafe for briefs |

Gemma 4's "small" E2B/E4B variants are 4.3 / 6.8 GB on disk (per-layer embeddings), so they
offer no memory advantage here. Set `[llm] model` in `config.toml` to switch; the brief panel
shows latency so you can compare on a real meeting.

## Requirements

- Apple Silicon Mac, macOS 14+ (tested on an M5, 16 GB)
- [LM Studio](https://lmstudio.ai) (for now) — `blaah models get qwen/qwen3.5-4b` fetches a good default
- Python 3.11+ and [uv](https://docs.astral.sh/uv/)

## Run

```bash
uv run blaah setup    # once: your name, and which model to use
uv run blaah          # start listening (or: uv run meeting-assistant)
```

First run downloads Parakeet (~1.2 GB) from Hugging Face. The LM Studio server and model
are started/loaded automatically if they aren't already. macOS will ask for microphone
access the first time. If you skip `setup`, the app asks your name when it starts.

Keys inside the app:

| Key | Action |
|---|---|
| `b` | Brief me now |
| `n` | Refresh the running notes now |
| `p` | Pause / resume the mic |
| `m` | Switch the model (remembered for next time) |
| `w` | Who's who — name a voice by hand (all its lines update) |
| `a` | Edit who's in the meeting |
| `r` | New meeting — save or delete this one first |
| `q` | Quit |

When you quit, the app refreshes the notes one last time, writes a wrap-up and asks for a
title. Every meeting is a folder of plain files under `meetings/` (set `[storage] meetings_dir`
to point it at an Obsidian vault or any folder):

```
meetings/2026-01-15_100000_weekly-sync/
  meeting.json     title, start/end, duration, word count, models used
  summary.md       wrap-up: TL;DR, decisions, action items, open questions, topics
  notes.md         last state of the running notes
  briefs.md        every brief that was generated
  transcript.md    readable transcript with timestamps and speaker names
  speakers.json    the voices heard, their names, and the evidence for each name
  transcript.jsonl the same, exact, for the app
```

A run that ends before anyone speaks (a crash, a dead battery) leaves an empty folder; the
next meeting tidies those away once they are 12 hours old. Folders with anything in them
are never touched.

```bash
uv run blaah --title "Weekly sync"     # name it up front (or when quitting)
uv run blaah list                      # past meetings
uv run blaah show 2026-01-15_1000      # a meeting's summary (prefix is enough)
uv run blaah ask "when is the cutover?" --meeting 2026-01-15   # answer from the transcript
```

`ask` answers from the meeting's transcript (plus its summary), cites the time of the line it
used, and says "Not discussed in this meeting." rather than guessing. Tested on a fabricated
meeting with planted facts (`scripts/make_fake_meeting.py` + `scripts/qa_eval.py`): Qwen3.5-4B
answered 13/13 correctly, including the trap question about a topic that was never raised,
at ~1.7 s per answer. Transcripts over ~2,200 words are sent as keyword-selected excerpts.

### Start it with Siri

Say **"Hey Siri, be my assistant"**. That runs the *Be my assistant* Shortcut, which opens
`Meeting Assistant.command` (in this folder) in Terminal. Double-clicking that file works too.
The Shortcut needs *Shortcuts › Settings › Advanced › Allow Running Scripts* switched on once.

### Test without a meeting

```bash
uv run blaah --file some_recording.wav
```

Runs the whole pipeline headless on an audio file in real time and prints the transcript,
notes and briefs (`--fast` feeds the audio as fast as possible). `--headless` does the same
with the live mic and no TUI. `--list-devices` shows input devices.

## Configure

Settings come in two layers:

- [`config.toml`](config.toml) — shared defaults, documented inline, part of the repo.
- `~/.config/blaah/config.toml` — **your** settings (your name, your model), never committed.
  `blaah setup`, `blaah models use` and the `m` key write it; any key from `config.toml`
  can go there to override it. Set `BLAAH_CONFIG` to keep it somewhere else.

The important bits:

- `[user] names` — your name and nicknames: when someone says one, you get a brief. Add
  spellings the transcriber tends to produce for it (it hears what it hears). Introductions
  ("Paul here", "this is Paul", "I'm Paul") don't count, and one mention gives one brief.
- `[llm] model` — which model writes notes and briefs (see *Choosing a model*).
- `[audio] input_device` — see below for capturing remote meetings.
- `[tracker] interval_seconds` — how often notes are refreshed.

### Choosing a model

Blaah works with any chat model you have in LM Studio.

```bash
uv run blaah models                       # what's downloaded, which one is in use
uv run blaah models use qwen/qwen3.5-9b   # switch (saved to your personal config)
uv run blaah models get qwen/qwen3.5-4b   # download one from the LM Studio catalog
```

Or press `m` during a meeting to switch on the spot. If the configured model isn't
downloaded, Blaah uses one that is (the loaded one, else a recommended one, else the
smallest) and tells you. Thinking models get their reasoning switched off for speed.
Smaller is faster: on a 16 GB Mac, `qwen/qwen3.5-4b` briefs in ~4 s, `qwen/qwen3.5-9b`
in ~8 s with somewhat better judgement (see *Performance*).

### Provider plugins

LM Studio is the built-in provider, and it's required for now. Other backends (Ollama,
OpenAI-compatible servers, …) can be added as plugins without changing Blaah: a plugin is a
Python package that registers a provider class under the `blaah.providers` entry point,

```toml
# in the plugin's pyproject.toml
[project.entry-points."blaah.providers"]
ollama = "blaah_ollama:OllamaProvider"
```

and implements the small interface in
[`meeting_assistant/providers.py`](meeting_assistant/providers.py) (`ensure_loaded`,
`complete`, `stream`, `list_models`, `download`). Install it next to Blaah and set
`[llm] provider = "ollama"`.

### Who said what (speaker names)

Every sentence gets a voiceprint (NVIDIA TitaNet-small, 40 MB, downloaded on first run;
~25 ms per sentence on the CPU). Voices are grouped as the meeting goes — *Speaker 1, 2, 3…* —
and get names from what people say:

| Cue | Example | Names |
|---|---|---|
| hand-off | "Dev, start with the cutover." | the next voice is Dev |
| introduction | "Hi all, Raj here" / "I'm Sarah" | this voice is Raj / Sarah |
| thanks | "Thanks, Karen." | the previous voice was Karen |

**Tell it who's in the meeting.** At start the app asks (pre-filled from your last meeting;
Enter to accept), or pass `--people "Shaun, Gavin"`. The transcriber mishears names it hasn't
learned — in a real Meet call one colleague's name came out as "given" and "Kivinus",
another's as "your shock" — so in cue positions a word that *sounds like* an attendee counts as them, while
"they have given us" is left alone. The list also goes into the notes, brief and summary
prompts, so they spell names right. Quick reactions ("Sorry.", "Cool.", "Sounds good") are
skipped when deciding who a hand-off went to.

Evidence adds up per voice and each name goes to one voice only, so a single misleading cue
("Karen, hold the comms" followed by someone else) is outvoted. Names can arrive late; earlier
lines are relabelled when they do. In calls captured with system audio, sentences that come
only from the mic count as evidence for *you*. To be recognised from your first word:

```bash
uv run blaah enroll                 # read a short paragraph for 20 s
```

Fixing names afterwards (no audio needed — every sentence keeps its voice id):

```bash
uv run blaah relabel --people "Shaun, Gavin" --meeting 2026-01-15_1000   # re-read cues
uv run blaah name --meeting 2026-01-15_1000                              # list voices + an example line
uv run blaah name S3 Leo --meeting 2026-01-15_1000                       # name one by hand
```

`[speakers] remember_voices = true` keeps colleagues' voiceprints between meetings
(`meetings/_voices/`, local files only). It's off by default — voiceprints are biometric
data, so turn it on only if the people you meet with are fine with that.

Limits, measured on a real four-person meeting (AMI corpus, crosstalk included): about 90 %
of speech time lands in the right voice. Sentences under ~2 s ("Yeah.", "Sure.") carry too
little voice to tell apart, so they're attributed to the speaker before them. A voice nobody
names stays *Speaker N*.

### Hearing the other side of a call (system audio)

The mic hears the room and whatever comes out of the speakers — fine without headphones.
With `[audio] system_audio` set (the default is `"all"`), the app *also* captures what the
Mac is playing through a native macOS audio tap (14.2+, no driver), mixed with the mic, so
remote participants are transcribed even on AirPods:

- `"all"` — everything the Mac plays
- `"brave"`, `"chrome"`, `"safari"`, `"zoom"`, `"teams"`, `"slack"` — just that app
  (any part of its bundle id); `""` turns it off

**Headphones and device switches.** Tested with Bluetooth earbuds:
- *The call side* is captured from the apps' audio wherever it plays, clocked by the Mac's
  built-in output (steady) rather than the Bluetooth device (which only clocks while playing
  and changes rate in call mode — that gave chopped or no audio). Parakeet transcribed it
  word for word with the buds on.
- *Your side:* if the default mic is a Bluetooth headset, the app records from the built-in
  mic instead (`prefer_builtin_mic`). Headset mics are call-quality, opening one switches the
  buds into call mode, and the earbuds' mic delivered near-silence (−72 dB) to anything but a call
  app. With headphones on, the built-in mic hears only you. If you turn that off and a
  Bluetooth mic isn't audible once in its first 15 s, it is swapped for the built-in one,
  with a note. (Other mics are left alone: a good USB mic is that quiet while you listen.)
- *Switching mid-meeting:* a watchdog reopens the mic when the default input changes or the
  stream stalls, and the system-audio tap relaunches if its clock device goes away.

To always use one particular mic, set `input_device` (e.g. `"MacBook Air Microphone"`).

It uses a small Swift helper in `helpers/audiotap/`, compiled on first run with the Xcode
Command Line Tools (`xcode-select --install`). The first time, macOS asks to allow
**System Audio Recording** for "Meeting Assistant audio tap" — if it was ever denied, enable
it under *System Settings › Privacy & Security › Screen & System Audio Recording*. The status
bar shows `system audio ✓` once real audio arrives. If the helper can't be built the app
falls back to mic only and says so.

## Project layout

```
meeting_assistant/
  audio.py        mic capture, mic + system-audio mix, FileCapture for tests
  speakers.py     voiceprints, voice clustering, name cues → who said what
  systemaudio.py  runs/builds the native audio-tap helper (helpers/audiotap/main.swift)
  transcriber.py  Parakeet streaming → finalised sentences + live text
  transcript.py   thread-safe transcript store, persisted to jsonl + md
  meeting.py      one meeting = one folder of plain files (meeting.json)
  summary.py      end-of-meeting wrap-up
  ask.py          questions about a past meeting (whole transcript, or excerpts if long)
  triggers.py     name detection with fuzzy matching + cooldown
  providers.py    model provider interface + plugin registry (entry point "blaah.providers")
  llm.py          the built-in LM Studio provider (native API, reasoning off, auto-load, fallback)
  naming.py       fix who-said-what in finished meetings (relabel, name by hand)
  tracker.py      background running-notes summariser (yields to briefs)
  brief.py        on-demand "what's going on / what to say"
  session.py      wires everything together, UI-agnostic
  app.py          Textual terminal UI
  notify.py       macOS notifications
config.toml       shared settings (yours: ~/.config/blaah/config.toml)
```

## Roadmap / ideas

- Global hotkey (works while Zoom is focused) — needs Accessibility permission
- Menu-bar app instead of a terminal UI
- Search across past meetings (plain grep works today; an index only if it ever gets slow)
- Briefs that recall earlier meetings on the same topic
- More providers as plugins (Ollama, OpenAI-compatible), making LM Studio optional
