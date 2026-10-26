
import wave
import math
import struct

sample_rate = 22050
duration = 3

with wave.open("sample_test_audio.wav", "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(sample_rate)

    for i in range(sample_rate * duration):
        value = int(5000 * math.sin(2 * math.pi * 200 * i / sample_rate))
        wav.writeframesraw(struct.pack("<h", value))

print("Valid WAV file created successfully!")
