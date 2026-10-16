# Jarvis: an AI agent for macOS

Jarvis is a personal assistant for my Mac. You type a request in plain English and it decides whether to answer, take an action (open an app, open a website, save a note), or search the web first.

It is powered by Claude (Anthropic's model) through the Claude API.

## What it can do

- Chat and remember the conversation
- Open any Mac app ("open spotify")
- Open websites and Google or YouTube searches in Chrome ("play starboy on youtube")
- Search the web and read pages to answer questions about current events
- Save notes as text files ("write a note called shopping.txt that says buy milk")

## How it works

Jarvis uses **tool calling**, the core pattern behind AI agents:

1. My message goes to Claude along with a list of tools it may use.
2. Claude replies either with text or with a request to call a tool.
3. My code runs the tool (for example `open -a Spotify`) and sends the result back.
4. Claude tells me what happened.

Web search and page reading run on Anthropic's servers. The Mac actions run as Python functions on my machine, and note files are restricted to one folder.

## What I learned building it

I first built Jarvis on a free local model (Llama 3.2, then Qwen 2.5 7B through Ollama). `jarvis.py` to `jarvis3.py` are those versions. I tested each version with the same set of requests and found:

- The 3B model could chat but invented fake tool calls.
- The 7B model handled 5 tools well (15 out of 15 in my tests), but with 7 tools it sometimes claimed to open apps without actually doing it, especially after a failed search.

So `jarvis4.py` switches to Claude, which handles all 7 tools reliably.

## Run it yourself

1. Get an API key from console.anthropic.com
2. Create a file called `.env` in this folder containing `ANTHROPIC_API_KEY=your-key` (it is in `.gitignore`, so it is never uploaded)
3. Install and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install anthropic python-dotenv
python jarvis4.py
```

## Roadmap

- [x] Chat with memory
- [x] Open apps and write files
- [x] Control Chrome and search the web
- [ ] Voice input and output
- [ ] Long-term memory (embeddings and vector search)
- [ ] Confirmation before risky actions, tests, demo video
