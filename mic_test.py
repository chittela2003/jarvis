"""Check that the microphone works and that Jarvis can hear "Hey Jarvis". Run: python mic_test.py"""
import numpy as np
import sounddevice as sd
from openwakeword.model import Model


def built_in_mic():
    for number, device in enumerate(sd.query_devices()):
        if "MacBook" in device["name"] and device["max_input_channels"] > 0:
            return number
    return None


mic = built_in_mic()
print("Using microphone:", sd.query_devices(mic if mic is not None else sd.default.device[0])["name"])
wake = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
print('Say "Hey Jarvis" 3 or 4 times in the next 10 seconds...\n')

loudest, best = 0.0, 0.0
with sd.InputStream(samplerate=16000, channels=1, dtype="int16", blocksize=1280, device=mic) as stream:
    for _ in range(125):  # 125 x 80 ms = 10 seconds
        audio, _ = stream.read(1280)
        level = float(np.abs(audio).mean())
        score = float(wake.predict(audio[:, 0])["hey_jarvis"])
        loudest, best = max(loudest, level), max(best, score)
        bar = "#" * min(40, int(level / 25))
        print(f"\r  sound {bar:40s}  wake word {score:.2f}", end="", flush=True)

print(f"\n\nLoudest sound: {loudest:.0f}   (under 50 means the microphone hears nothing)")
print(f"Best 'Hey Jarvis' score: {best:.2f}   (Jarvis reacts above 0.40)")
