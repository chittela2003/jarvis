"""Make Jarvis start by itself when you log in, and keep running in the background in a hidden window.

    python install_background.py           turn background Jarvis on (it starts straight away)
    python install_background.py --remove  turn it off again
"""
import os
import plistlib
import subprocess
import sys
from pathlib import Path

LABEL = "com.jarvis.assistant"  # the name macOS uses for this background job
FOLDER = Path(__file__).resolve().parent
AGENT = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
DOMAIN = f"gui/{os.getuid()}"  # "the logged-in user", so Jarvis can use the mic and speakers
LOG = FOLDER / "jarvis.log"
DESKTOP_ICON = Path.home() / "Desktop" / "Jarvis.command"

# Stop any copy that is already running in the background
subprocess.run(["launchctl", "bootout", f"{DOMAIN}/{LABEL}"], capture_output=True)

if "--remove" in sys.argv:
    AGENT.unlink(missing_ok=True)
    print("Background Jarvis is off. It will not start when you log in.")
    sys.exit()

AGENT.parent.mkdir(exist_ok=True)
with AGENT.open("wb") as file:
    plistlib.dump({
        "Label": LABEL,
        # Run Jarvis inside a hidden Terminal (-g = in the background, -j = hidden), because
        # Terminal already has microphone permission and a background Python does not
        "ProgramArguments": ["/usr/bin/open", "-g", "-j", "-a", "Terminal", str(FOLDER / "start_jarvis.command")],
        "RunAtLoad": True,  # start when you log in
    }, file)
subprocess.run(["launchctl", "bootstrap", DOMAIN, str(AGENT)], check=True)

# The Desktop icon now shows what background Jarvis is doing, instead of starting a second Jarvis
DESKTOP_ICON.write_text(f"#!/bin/zsh\necho 'Background Jarvis log (close this window any time - Jarvis keeps running)'\ntail -n 40 -f '{LOG}'\n")
DESKTOP_ICON.chmod(0o755)

print("Background Jarvis is on. It is starting now and will start by itself every time you log in.")
print("Say \"Hey Jarvis\" in about 15 seconds. Double-click Jarvis.command on the Desktop to see what it hears.")
print("Jarvis runs inside Terminal, so if you quit Terminal completely (Cmd+Q), Jarvis stops until you log in again.")
