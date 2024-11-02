print('starting...')

import device_manager_windows

assert device_manager_windows.init_windows(), 'startup failed, ensure virtual devices are installed properly'

import re
import time
import queue
import argparse
import platform
import threading
import numpy as np

if platform.system() == 'Windows':
    import pyaudiowpatch as pyaudio
else:
    import pyaudio

# from scipy.signal import resample

import zen

parser = argparse.ArgumentParser()
# TODO: instead of cpu, pass device name (cpu, cuda:0, cuda:1, mps, etc.)
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--input')
parser.add_argument('--output')
parser.add_argument('--buffer-size', type=int, default=127)
parser.add_argument('--samples-per-io', type=int, default=4096)
args = parser.parse_args()

assert 0 < args.buffer_size < 128, 'buffer_size must be greater than 0 and less than 128'
assert 0 < args.samples_per_io < 1024 * 128, 'samples_per_io must be greater than 0 and less than 1024 * 128'

use_gpu = not args.cpu

frames_per_buffer = args.samples_per_io
channels = 2
samplerate = 48000
# this still works as low as 1024 * 50 but the delay is only marginally lower
# at 1024 * 1023 and samplerate = 48000 the delay is about 1024 * 1023 / 48000 = 21.824 seconds
# at 1024 * 50 it should be ~1sec delay but getting about ~9sec delay
buffer_size = 1024 * args.buffer_size

dev_in_pattern = args.input
if not dev_in_pattern:
    dev_in_pattern = 'BlackHole|CABLE Input'

dev_out_pattern = args.output
if not dev_out_pattern:
    dev_out_pattern = 'Speakers|Headphones'

# volume between 0..1
#volume = 0.5

# Initialize PyAudio
pa = pyaudio.PyAudio()

# Find the index of the VB Cable / Blackhole device
dev_in = None
dev_out = None
devices_by_name = {}
devices_by_index = {}
for i in range(pa.get_device_count()):
    dev = pa.get_device_info_by_index(i)
    devices_by_index[dev['index']] = dev
    devices_by_name[dev['name']] = dev

for idx, dev in devices_by_index.items():
    if dev['maxInputChannels'] > 0 and (str(idx) == dev_in_pattern or re.search(dev_in_pattern, dev['name'], re.IGNORECASE)):
        if dev_in:
            print(f'skipping additional matching input device: {dev["index"]}, {dev["name"]}')
        else:
            dev_in = dev

    if dev['maxOutputChannels'] > 0 and (str(idx) == dev_out_pattern or re.search(dev_out_pattern, dev['name'], re.IGNORECASE)):
        if dev_out:
            print(f'skipping additional matching output device: {dev["index"]}, {dev["name"]}')
        else:
            dev_out = dev

assert dev_in, f'input device not found: {dev_in_pattern}'
assert dev_out, f'output device not found: {dev_out_pattern}'

print('-' * 80)
print(f'input device: {dev_in["index"]} {dev_in["name"]} ({dev_in["maxInputChannels"]} channels, samplerate {dev_in["defaultSampleRate"]})')
print(f'output device: {dev_out["index"]} {dev_out["name"]} ({dev_out["maxOutputChannels"]} channels, samplerate {dev_out["defaultSampleRate"]})')
print(f'preferred processor: {"cpu" if args.cpu else "gpu"}')
print(f'buffer size: {buffer_size}')
print(f'samples per io: {frames_per_buffer}')

input_queue = queue.Queue()
output_queue = queue.Queue()

def zen_thread():
    print('initializing')
    demixer = zen.ZenDemixer(use_gpu=use_gpu)

    back_buffer = np.zeros((channels, 0), dtype=np.float32)

    assert buffer_size <= demixer.chunk_size, 'buffer_size must be less than or equal to demixer.chunk_size'

    while True:
        input_buffer = input_queue.get()
        # take the last chunk_size samples
        if len(input_buffer) < demixer.chunk_size:
            remainder = demixer.chunk_size - len(input_buffer)
            input_buffer = np.concatenate((back_buffer[:, -remainder:], input_buffer), axis=1)

        print('demixing')
        output_buffer = demixer.demix(input_buffer) #, buffer_size=demixer.chunk_size)
        print('writing output')
        output_queue.put(output_buffer[:, -buffer_size:])

        back_buffer = back_buffer[:, -demixer.chunk_size:]

def audio_output_thread(output_stream):
    while True:
        
        if output_queue.empty():
            # TODO: output some default soundwave
            continue

        print('got demixed output chunk')
        output_data = output_queue.get()
        output_data_compact = output_data.transpose().flatten() # output_data.reshape((channels, -1)).transpose().flatten()
        print('writing output')
        output_stream.write(output_data_compact.tobytes())
        print('output written')


def process_forever(input_stream, output_stream):
    thread = threading.Thread(target=audio_output_thread, args=(output_stream,))
    thread.start()

    input_buffer = np.zeros((channels, 0), dtype=np.float32)

    while True:
        # len(input_data) == (frames_per_buffer * channels * sizeof(float))
        input_data = input_stream.read(frames_per_buffer)

        # len(audio_data) == (frames_per_buffer * channels)
        audio_data_compact = np.array(np.frombuffer(input_data, dtype=np.float32))

        # demix_samplerate = 44100
        # audio_data_compact = resample(audio_data_compact, int(len(audio_data_compact) * demix_samplerate / samplerate))

        # audio_data_compact has interleaved samples for each channel, convert to separate channels
        audio_data = audio_data_compact.reshape((-1, channels)).transpose()
        # audio_data = np.array([audio_data_compact[::2], audio_data_compact[1::2]])

        if not audio_data.any():
            print('empty buffer')
            #continue

        input_buffer = np.concatenate((input_buffer, audio_data), axis=1)

        if input_buffer.shape[1] >= buffer_size:
            print('input buffer filled, processing')
            input_queue.put(input_buffer[:, :buffer_size])
            input_buffer = input_buffer[:, buffer_size:]



def main():
    thread = threading.Thread(target=zen_thread)
    thread.start()

    try:
        # Get the default WASAPI loopback device
        # loopback_device = p.get_default_wasapi_loopback()

        # Open an input stream to capture audio from the loopback device
        # NOTE: the device must not be muted or have volume 0 in the operating system settings
        input_stream = pa.open(format=pyaudio.paFloat32,
                              channels=channels,
                              rate=samplerate,
                              input=True,
                              input_device_index=dev_in['index'],
                              frames_per_buffer=frames_per_buffer)

        # Open an output stream to play the processed audio
        output_stream = pa.open(format=pyaudio.paFloat32,
                               channels=channels,
                               rate=samplerate,
                               output=True,
                               frames_per_buffer=frames_per_buffer,
                               output_device_index=dev_out['index'])

        print("Starting audio processing. Press Ctrl+C to stop.")

        process_forever(input_stream, output_stream)

    except KeyboardInterrupt:
        print("\nStopping audio processing.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Cleanup
        if 'input_stream' in locals():
            input_stream.stop_stream()
            input_stream.close()
        if 'output_stream' in locals():
            output_stream.stop_stream()
            output_stream.close()
        pa.terminate()

if __name__ == "__main__":
    main()
