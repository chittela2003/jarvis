#!/bin/zsh
# Starts Jarvis from Terminal (which has microphone permission), detached from this window,
# so closing the window or Terminal does not stop Jarvis
cd ~/jarvis
if pgrep -f "jarvis11.py" > /dev/null; then
    echo "Jarvis is already running."
    exit 0
fi
nohup .venv/bin/python -u jarvis11.py >> jarvis.log 2>&1 &
disown
echo "Jarvis started in the background. You can close this window."
