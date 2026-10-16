# Jarvis: a local AI agent for macOS

Jarvis is a personal assistant that runs entirely on my Mac. You type a request in plain English and it decides whether to answer or take an action, like opening an app or saving a note.

It uses a local language model (Qwen 2.5 7B through Ollama), so it is free and nothing leaves my computer.

## What it can do

- Chat and remember the conversation
- Open any Mac app ("open google chrome")
- Save notes as text files ("write a note called shopping.txt that says buy milk")

## How it works

Jarvis uses **tool calling**, the core pattern behind AI agents:

1. My message goes to the model along with a list of tools (Python functions) it may use.
2. The model replies either with text or with a request to call a tool.
3. My code runs the tool and sends the result back to the model.
4. The model tells me what happened.

The model never runs anything itself. My code decides what actually executes, and note files are restricted to one folder.

## Run it yourself

1. Install [Ollama](https://ollama.com) and run `ollama pull qwen2.5:7b`
2. Set up Python:
   python3 -m venv .venv
   source .venv/bin/activate
   pip install ollama
3. Run `python jarvis2.py`

## Roadmap

- [x] Chat with memory
- [x] Open apps and write files
- [ ] Control Chrome (Playwright)
- [ ] Voice input and output
- [ ] Long-term memory (embeddings and vector search)
- [ ] Confirmation before risky actions, tests, demo video
