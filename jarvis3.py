import subprocess
from pathlib import Path
from urllib.parse import quote_plus
import ollama

MODEL = "qwen2.5:7b"


def open_app(name: str) -> str:
    """
    Open any app installed on this Mac.

    Args:
        name: The app name, for example Spotify, WhatsApp, Notes or Calculator
    """
    result = subprocess.run(["open", "-a", name], capture_output=True, text=True)
    if result.returncode == 0:
        return f"Opened {name}"
    return f"Could not open {name}: {result.stderr.strip()}"


def write_note(filename: str, text: str) -> str:
    """
    Save a text note as a file in the notes folder.

    Args:
        filename: The file name, for example shopping.txt
        text: What to write in the note
    """
    folder = Path.home() / "jarvis" / "notes"
    folder.mkdir(exist_ok=True)
    path = folder / Path(filename).name
    path.write_text(text)
    return f"Saved note to {path}"


def open_in_chrome(url: str) -> str:
    """
    Open a web page in Google Chrome.

    Args:
        url: The address to open, for example https://github.com
    """
    if not url.startswith("http"):
        url = "https://" + url
    subprocess.run(["open", "-a", "Google Chrome", url])
    return f"Opened {url} in Chrome. You cannot see what is on the page."


def search_google(query: str) -> str:
    """
    Search Google in Chrome.

    Args:
        query: What to search for
    """
    return open_in_chrome("https://www.google.com/search?q=" + quote_plus(query))


def search_youtube(query: str) -> str:
    """
    Search YouTube in Chrome. Use this when the user wants to play, watch or find a video or song.

    Args:
        query: What to search for
    """
    return open_in_chrome("https://www.youtube.com/results?search_query=" + quote_plus(query))


skills = [open_app, write_note, open_in_chrome, search_google, search_youtube]
tools = {skill.__name__: skill for skill in skills}

history = [
    {"role": "system", "content": "You are Jarvis, a helpful assistant on the user's Mac. Use a tool only when the user asks you to do something on the computer, such as opening an app, saving a note, opening a website, or searching Google or YouTube. For everything else, just reply in plain words. After using a tool, say only what you did. Never make up what a web page shows, and never claim something is playing."}
]

while True:
    text = input("You: ")
    history.append({"role": "user", "content": text})

    reply = ollama.chat(model=MODEL, messages=history, tools=skills)
    history.append(reply.message)

    if reply.message.tool_calls:
        for call in reply.message.tool_calls:
            function = tools[call.function.name]
            result = function(**call.function.arguments)
            print("  [Jarvis used", call.function.name, "->", result + "]")
            history.append({"role": "tool", "content": result})
        reply = ollama.chat(model=MODEL, messages=history)
        history.append(reply.message)

    print("Jarvis:", reply.message.content)