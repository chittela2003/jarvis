import subprocess
from pathlib import Path  # NEW
import ollama


def open_app(name: str) -> str:
    """
    Open an app on this Mac.

    Args:
        name: The app name, for example Safari, Notes or Calculator
    """
    result = subprocess.run(["open", "-a", name], capture_output=True, text=True)
    if result.returncode == 0:
        return f"Opened {name}"
    return f"Could not open {name}: {result.stderr.strip()}"


# NEW: Jarvis's second skill
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


tools = {"open_app": open_app, "write_note": write_note}  # NEW: added write_note

history = [
    # NEW: the instruction now mentions write_note too
    {"role": "system", "content": "You are Jarvis, a helpful assistant on the user's Mac. Use open_app only when the user asks you to open an app, and write_note only when they ask you to save or write a note. For everything else, just reply in plain words."}
]

while True:
    text = input("You: ")
    history.append({"role": "user", "content": text})

    # NEW: added write_note to the list
    reply = ollama.chat(model="qwen2.5:7b", messages=history, tools=[open_app, write_note])
    history.append(reply.message)

    if reply.message.tool_calls:
        for call in reply.message.tool_calls:
            function = tools[call.function.name]
            result = function(**call.function.arguments)
            print("  [Jarvis used", call.function.name, "->", result + "]")
            history.append({"role": "tool", "content": result})
        reply = ollama.chat(model="qwen2.5:7b", messages=history)
        history.append(reply.message)

    print("Jarvis:", reply.message.content)