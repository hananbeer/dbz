import pyaudio
import wave
import numpy as np
import math

CHUNK = 1024
FORMAT = pyaudio.paInt16
# CHANNELS = 2
# RATE = 44100 # 48000
RECORD_SECONDS = 50
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

# Find the index of the Blackhole device
dev_in = None
dev_out = None
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    #print(dev)
    if "BlackHole" in dev['name']:
        dev_in = dev
    if "Speakers" in dev['name']:
        dev_out = dev

print('input', dev_in)
print('output', dev_out)
stream_in = p.open(format=FORMAT,
                channels=dev_in['maxInputChannels'],
                rate=int(dev_in['defaultSampleRate']),
                input=True,
                input_device_index=dev_in['index'],
                frames_per_buffer=CHUNK)

stream_out = p.open(format=FORMAT,
                channels=dev_out['maxOutputChannels'],
                rate=int(dev_out['defaultSampleRate']),
                output=True,
                output_device_index=dev_out['index'],
                frames_per_buffer=CHUNK)

print("* recording")

frames = []
has_data = None
for i in range(0, int(dev_in['defaultSampleRate'] / CHUNK * RECORD_SECONDS)):
    data = stream_in.read(CHUNK)
    frames.append(data)
    if not np.frombuffer(data, dtype=np.int16).any():
        if has_data != False:
            print("no data")
            has_data = False
    elif has_data != True:
        print("data!!")
        has_data = True
        #data = [int(data[i] * math.sin(i)) for i in range(len(data))]
    stream_out.write(data)

print("* done recording")

stream_in.stop_stream()
stream_in.close()
stream_out.stop_stream()
stream_out.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(dev_in['maxInputChannels'])
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(int(dev_in['defaultSampleRate']))
wf.writeframes(b''.join(frames))
wf.close()