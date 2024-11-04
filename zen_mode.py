###
### parse args
###

import argparse

parser = argparse.ArgumentParser()
# TODO: instead of cpu, pass device name (cpu, cuda:0, cuda:1, mps, etc.)
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--virtual-device')
parser.add_argument('--output-device')
parser.add_argument('--model-size', type=int, choices=[5, 6, 7, 8, 9, 10, 11], default=5)
parser.add_argument('--buffer-size', type=int, default=None)
parser.add_argument('--samples-per-io', type=int, default=1024)
parser.add_argument('--volume-music', type=int, default=100, help='music volume level')
parser.add_argument('--volume-vocals', type=int, default=0, help='vocals volume level')
args = parser.parse_args()

###
### initiate startup, install virtual devices if necessary
###

print('initializing...')

import platform

if platform.system() == 'Windows':
    import pyaudiowpatch as pyaudio
    import device_manager_windows as devman
else:
    import pyaudio
    import device_manager_mac as devman

try:
    assert devman.startup(), 'startup failed, ensure virtual devices are installed properly'
except Exception as e:
    print(f'startup failed: {e}')
    exit(1)

###
### start app
###

import re
import time
import queue
import atexit
import threading
import traceback
import numpy as np

import zen_demixer

use_gpu = not args.cpu

model_size = args.model_size
buffer_base = args.buffer_size if args.buffer_size else (2 ** model_size - 1)
frames_per_buffer = args.samples_per_io

volume_music = args.volume_music / 100.0
volume_vocals = args.volume_vocals / 100.0

assert 0 < buffer_base < 2 ** model_size, 'buffer_size must be greater than 0 and less than 128'
assert 0 < frames_per_buffer < 1024 * 128, 'samples_per_io must be greater than 0 and less than 1024 * 128'

channels = 2

# this still works as low as 1024 * 50 but the delay is only marginally lower
# at 1024 * 1023 and samplerate = 48000 the delay is about 1024 * 1023 / 48000 = 21.824 seconds
# at 1024 * 50 it should be ~1sec delay but getting about ~9sec delay
buffer_size = 1024 * buffer_base

dev_in_pattern = args.virtual_device
if not dev_in_pattern:
    dev_in_pattern = 'BlackHole|CABLE Input'

dev_out_pattern = args.output_device
if not dev_out_pattern:
    dev_out_pattern = 'Speakers|Headphones'

print('dev_out_pattern', dev_out_pattern)

# this must be called before initializing PyAudio
devman.set_virtual_audio_device_as_default()
atexit.register(devman.restore_default_audio_device)

# Initialize PyAudio
pa = pyaudio.PyAudio()

# Find the index of the VB Cable / Blackhole device
dev_in = None
dev_out = None
for idx in range(pa.get_device_count()):
    dev = pa.get_device_info_by_index(idx)
    if dev['maxInputChannels'] > 0 and (str(idx) == dev_in_pattern or re.search(dev_in_pattern, dev['name'], re.IGNORECASE)):
        if dev_in:
            print(f'!! skipping additional matching input device: {dev["index"]}, {dev["name"]}')
        else:
            dev_in = dev

    if dev['maxOutputChannels'] > 0 and (str(idx) == dev_out_pattern or re.search(dev_out_pattern, dev['name'], re.IGNORECASE)):
        if dev_out:
            print(f'!! skipping additional matching output device: {dev["index"]}, {dev["name"]}')
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
print('=' * 80)

input_queue = queue.Queue()
output_queue = queue.Queue()

def print_time(stime, msg, end='\n'):
    print(f'\r{msg}: %.2fs' % (time.perf_counter() - stime), end=end)

def demixer_thread(demixer):
    assert buffer_size <= demixer.chunk_size, 'buffer_size must be less than or equal to demixer.chunk_size'

    back_buffer = np.zeros((channels, 0), dtype=np.float32)

    try:
        while True:
            input_buffer = input_queue.get()
            # take the last chunk_size samples
            if len(input_buffer) < demixer.chunk_size:
                remainder = demixer.chunk_size - len(input_buffer)
                input_buffer = np.concatenate((back_buffer[:, -remainder:], input_buffer), axis=1)

            stime = time.perf_counter()
            output_buffer = demixer.demix(input_buffer, volume_music=volume_music, volume_vocals=volume_vocals) #, buffer_size=demixer.chunk_size)
            print_time(stime, 'demixed audio', end='\r')

            output_queue.put(output_buffer[:, -buffer_size:])
            back_buffer = back_buffer[:, -demixer.chunk_size:]
    except Exception as e:
        print(f'demixer thread error')
        print(traceback.format_exc())

def audio_output_thread(output_stream):
    try:
        while True:
            # if output_queue.empty():
            #     # TODO: output some default soundwave
            #     continue

            output_data = output_queue.get()
            output_data_compact = output_data.transpose().flatten() # output_data.reshape((channels, -1)).transpose().flatten()

            # writing to output_stream is blocking, so it takes len(output_data_compact) / samplerate seconds to play
            # stime = time.perf_counter()
            output_stream.write(output_data_compact.tobytes())
            # print_time(stime, 'played processed audio')
    except Exception as e:
        print(f'audio output thread error')
        print(traceback.format_exc())


def process_forever(input_stream, output_stream):
    # NOTE: can open input_stream to start filling back_buffer while model is loading
    print('loading model...', end='\r')
    stime = time.perf_counter()
    demixer = zen_demixer.ZenDemixer(use_gpu=use_gpu, segment_size=2 ** model_size)
    print_time(stime, 'model loaded')

    zthread = threading.Thread(target=demixer_thread, args=(demixer,), daemon=True)
    zthread.start()

    athread = threading.Thread(target=audio_output_thread, args=(output_stream,), daemon=True)
    athread.start()

    input_buffer = np.zeros((channels, 0), dtype=np.float32)

    prev_empty = None
    while True:
        if not zthread.is_alive():
            print('zen thread stopped unexpectedly, halting')
            break

        if not athread.is_alive():
            print('audio output thread stopped unexpectedly, halting')
            break

        # len(input_data) == (frames_per_buffer * channels * sizeof(float))
        input_data = input_stream.read(frames_per_buffer)

        # len(audio_data) == (frames_per_buffer * channels)
        audio_data_compact = np.frombuffer(input_data, dtype=np.float32)

        # example how to resample, probably not needed
        # demix_samplerate = 44100
        # audio_data_compact = resample(audio_data_compact, int(len(audio_data_compact) * demix_samplerate / samplerate))

        # audio_data_compact has interleaved samples for each channel, convert to separate channels
        audio_data = audio_data_compact.reshape((-1, channels)).transpose()

        if not audio_data.any():
            if not prev_empty:
                print('empty buffer')
                prev_empty = True
            #continue
        else:
            if prev_empty:
                print('got audio data')
                prev_empty = False

        input_buffer = np.concatenate((input_buffer, audio_data), axis=1)

        if input_buffer.shape[1] >= buffer_size:
            # print('buffer full, processing')
            input_queue.put(input_buffer[:, :buffer_size])
            input_buffer = input_buffer[:, buffer_size:]


def main():
    input_stream = None
    output_stream = None

    try:
        # Open an input stream to capture audio from the loopback device
        # NOTE: the device must not be muted or have volume 0 in the operating system settings
        input_stream = pa.open(format=pyaudio.paFloat32,
                              channels=channels,
                              rate=int(dev_in['defaultSampleRate']),
                              input=True,
                              input_device_index=dev_in['index'],
                              frames_per_buffer=frames_per_buffer)

        # Open an output stream to play the processed audio
        output_stream = pa.open(format=pyaudio.paFloat32,
                               channels=channels,
                               rate=int(dev_in['defaultSampleRate']), # TODO: using same sample rate here, assuming it will work but need to fix - either resample or use dev_out['defaultSampleRate']
                               output=True,
                               frames_per_buffer=frames_per_buffer,
                               output_device_index=dev_out['index'])

        print("[starting audio processing. press CTRL+C to stop.]")

        process_forever(input_stream, output_stream)
    except KeyboardInterrupt:
        print("\nstopping audio processing.")
    except Exception as e:
        print(f"an error occurred")
        print(traceback.format_exc())
    finally:
        # cleanup
        # this will also be called atexit callback, but it seems it is not always called? so just call here too
        devman.restore_default_audio_device()

        # TODO: do this in a thread.join() of all other threads or find a better way
        # to handle crashes/force quits
        if input_stream:
            try:
                input_stream.stop_stream()
                input_stream.close()
            except:
                pass

        if output_stream:
            try:
                output_stream.stop_stream()
                output_stream.close()
            except:
                pass

        pa.terminate()

        time.sleep(3)
        exit(1)

main()
