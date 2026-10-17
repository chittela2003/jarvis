"""Record a few short clips of your voice so Jarvis's wake word can be tuned to you.
Run: python record_samples.py   (the clips stay on your Mac, in the voice_samples folder)"""
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

RATE = 16000
FOLDER = Path(__file__).parent / "voice_samples"
CLIPS = (
    [("hey_jarvis", "Hey Jarvis")] * 6
    + [("jarvis_command", "Jarvis, open Spotify")] * 4
    + [("other", "I am going to make some tea now"), ("other", "What time is the cricket match tonight"),
       ("other", "Can you pass me the charger please")]
)


def built_in_mic():
    for number, device in enumerate(sd.query_devices()):
        if "MacBook" in device["name"] and device["max_input_channels"] > 0:
            return number
    return None


FOLDER.mkdir(exist_ok=True)
print(f"You will say {len(CLIPS)} short phrases. Speak in your normal voice, like you would to Jarvis.")
for number, (kind, phrase) in enumerate(CLIPS, start=1):
    input(f'\n{number}/{len(CLIPS)}  Press Enter, then say:  "{phrase}"')
    audio = sd.rec(int(3 * RATE), samplerate=RATE, channels=1, dtype="int16", device=built_in_mic())
    sd.wait()  # records for 3 seconds
    with wave.open(str(FOLDER / f"{kind}_{number:02d}.wav"), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(RATE)
        file.writeframes(audio.tobytes())
    print("  saved" if np.abs(audio).mean() > 50 else "  saved (that was very quiet - speak a bit louder next time)")

print("\nDone! Tell Claude, and it will tune Jarvis to your voice.")
