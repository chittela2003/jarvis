"""Teach Jarvis what your voice sounds like. Run this once: python enroll_voice.py"""
import ssl
import urllib.request
from pathlib import Path

import certifi
import numpy as np
from openwakeword.utils import download_models
import sherpa_onnx
import sounddevice as sd

RATE = 16000
FOLDER = Path(__file__).parent
MODEL = FOLDER / "models" / "speaker.onnx"
MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
             "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx")
# Say the same thing you will say to Jarvis: matching works best on the wake word itself
SENTENCES = ["Hey Jarvis"] * 5

print('Downloading the "Hey Jarvis" wake word model...')
download_models(["hey_jarvis"])

if not MODEL.exists():
    print("Downloading the voice recognition model (40 MB)...")
    MODEL.parent.mkdir(exist_ok=True)
    # certifi supplies the security certificates that this Python install is missing
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(MODEL_URL, context=context) as response:
        MODEL.write_bytes(response.read())


def built_in_mic():
    # Always use the Mac's own microphone, even when AirPods are connected: AirPods
    # microphones often fail to open, and your voiceprint is recorded on this mic
    for number, device in enumerate(sd.query_devices()):
        if "MacBook" in device["name"] and device["max_input_channels"] > 0:
            return number
    return None  # no built-in mic found, so use the default one


speaker = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(MODEL)))
vectors = []
for sentence in SENTENCES:
    input(f'\nPress Enter, then read this out loud in your normal voice:\n    "{sentence}"')
    audio = sd.rec(3 * RATE, samplerate=RATE, channels=1, dtype="float32", device=built_in_mic())
    sd.wait()  # records for 3 seconds
    audio = audio[:, 0]
    # Keep only the part where you spoke (plus a little around it), not the silence
    loud = np.flatnonzero(np.abs(audio) > np.abs(audio).max() * 0.1)
    if len(loud) == 0:
        print("I didn't hear anything - check the microphone and try again.")
        continue
    audio = audio[max(0, loud[0] - RATE // 4): loud[-1] + RATE // 4]
    stream = speaker.create_stream()
    stream.accept_waveform(RATE, audio)
    stream.input_finished()
    vector = np.array(speaker.compute(stream))
    vectors.append(vector / np.linalg.norm(vector))
    print("Got it.")

voiceprint = np.mean(vectors, axis=0)
np.save(FOLDER / "voiceprint.npy", voiceprint / np.linalg.norm(voiceprint))
print("\nDone! Jarvis now knows your voice. Start Jarvis and say \"Hey Jarvis\".")
