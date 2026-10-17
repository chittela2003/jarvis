# Jarvis: a voice-controlled AI agent for macOS

Jarvis is a personal assistant that runs in the background on my Mac. I say **"Hey Jarvis"** (or clap twice), give a command out loud, and it decides whether to answer, take an action on the Mac, search the web, or remember something about me. It only obeys my voice.

The thinking is done by Claude (Anthropic's model) through the Claude API. Hearing, voice recognition and memory search run locally on the Mac.

## What it can do

- **Wake up hands-free:** say "Hey Jarvis" or clap twice. Clap three times to make it stop listening.
- **Only respond to my voice:** a voiceprint check ignores other people.
- **Talk back:** replies are spoken aloud, and important events ("I'm ready for your command", "I'm done", errors) also appear as pop-up notifications.
- **Control the Mac:** open any app, set the volume, mute, dark mode, Wi-Fi, Bluetooth, turn the screen off, and run any Shortcut from the Shortcuts app.
- **Use Chrome:** open websites and Google or YouTube searches.
- **Search the web** and read pages to answer questions about current events.
- **Remember things** about me between sessions, and look them up by meaning.
- **Save and delete notes**, with deleted notes kept in a recovery folder.

## How it works

```
microphone ──► wake word model ("Hey Jarvis")  ┐
           └─► clap counter (2 or 3 claps)     ┴─► record until I stop talking
                                                       │
                       voiceprint check (is this me?) ◄┘
                                                       │
                            Whisper speech-to-text ◄───┘
                                                       │
                   Claude + tools (the agent loop) ◄───┘ ──► macOS `say` + notification
```

**The agent loop (tool calling).** My words go to Claude with a list of tools, which are Python functions such as `open_app` or `set_volume`. Claude either answers or asks for a tool to be run. My code runs it and sends the result back, and Claude replies. Web search and page reading are Anthropic server tools; everything else runs on my Mac.

**Long-term memory (RAG).** Each remembered fact is stored with an embedding (384 numbers from the `bge-small-en` model) that captures its meaning. When a question is personal, Claude calls `recall`, which embeds the question and returns the 5 closest facts, so Jarvis never has to read its whole memory.

**Voice lock.** A speaker-recognition model turns a clip of speech into a 512-number voiceprint. My voiceprint was made from my own "Hey Jarvis" recordings; a command only runs if it matches by at least 50%.

## Safety design

- **Risky actions need my confirmation**, enforced in code (not just in the prompt): deleting a note, or turning Wi-Fi or Bluetooth off. `ASK_BEFORE_RISKY` in `jarvis11.py` switches this off.
- **Undoable deletes:** notes are moved to `notes/deleted/`, never erased.
- **Inputs are never trusted:** note filenames are stripped of folder paths so they can't escape the notes folder, and volume is clamped to 0-100.
- **Prompt injection:** text from web pages is treated as information only, never as instructions.
- **Private data stays private:** the API key, memories, voiceprint and voice recordings are all in `.gitignore`.

## What I learned building it

The project is in stages; each `jarvisN.py` is one step, so the history shows how it grew.

| File | Stage |
|---|---|
| `jarvis.py` – `jarvis3.py` | Local models through Ollama: chat, then tools, then Chrome |
| `jarvis4.py` | Switched to Claude, added web search |
| `jarvis5.py`, `jarvis6.py` | Speaks (macOS `say`) and listens (Whisper) |
| `jarvis7.py`, `jarvis8.py` | Long-term memory, then search by meaning (RAG) |
| `jarvis9.py`, `jarvis10.py` | Confirmations for risky actions, Mac settings control |
| `jarvis11.py` | Wake word, voice lock, claps, spoken alerts, background mode |

Things I found by testing rather than guessing:

- **Small local models struggle with many tools.** Llama 3.2 (3B) invented fake tool calls. Qwen 2.5 (7B) chose the right tool 15 out of 15 times with 5 tools, but with 7 tools it sometimes *said* it had opened an app without doing it. That is why I moved to Claude.
- **Tool descriptions matter as much as code.** When `open_app`'s example list was "Safari, Notes, Calculator", the model refused to open Spotify because it assumed only built-in apps worked.
- **Accents matter for voice.** In tests with computer-generated voices, a generic wake-word model scored 0.99 on American voices but 0.44 on an Indian English voice, and Whisper heard "Jarvis" as "John Lewis". So I measured on recordings of my real voice and tuned the thresholds to it.
- **A bad enrollment breaks the voice lock.** My first voiceprint, made from 5-second clips that were mostly silence, matched my real voice only 33-43%. Rebuilding it from real "Hey Jarvis" clips raised that to 76-89%, while other voices stayed around 20%.
- **macOS blocks the microphone for background processes** unless they start from an app that already has permission, so background mode starts Jarvis from Terminal and then detaches it.

## Measuring it

`eval_tools.py` is an automated test: it sends 31 requests (`evals/tool-choice/cases.json`) to Claude with Jarvis's real instructions and tools, and checks which tool Claude picks. The tools are never actually run. It includes requests that need **no** tool (a joke, a maths question), so a Jarvis that used tools for everything would fail. Before trusting it, I checked the test itself: a fake "perfect" model scored 100% and a fake "never uses tools" model scored 12%.

- **Result: 31/31** requests got the right tool (on Claude Opus 5.5). Per-case results are in `evals/tool-choice/baseline/results.jsonl`.
- **It also found a cost bug.** Every request was paying full price for the same ~8,700 tokens of instructions and tool descriptions; none were read from the prompt cache. Adding a cache marker after the instructions cut the cost of a command from about **$0.044 to $0.002–0.003** (measured on 3 real requests), roughly 94% cheaper. The first command after about 5 idle minutes still pays the full price once.

Known limits: a very similar voice or a recording of me can pass the voice lock, and claps are not voice-checked (but the command that follows is).

## Run it yourself

1. Get an API key from console.anthropic.com and put it in a file called `.env` in this folder:
   `ANTHROPIC_API_KEY=your-key`
2. Install:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Teach Jarvis your voice: `python enroll_voice.py`
4. Run it: `python jarvis11.py`, or `python install_background.py` to start it automatically at login.

Bluetooth control also needs `brew install blueutil`.

## Roadmap

- [x] Chat with memory
- [x] Open apps and write files
- [x] Control Chrome and search the web
- [x] Voice input and output
- [x] Long-term memory (embeddings and vector search)
- [x] Confirmation before risky actions
- [x] Wake word, voice lock and clap control
- [x] Automated accuracy test (31/31 right tool) and a prompt-caching cost fix
- [ ] Demo video
