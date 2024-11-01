import zen
import time
import queue
import threading
import numpy as np

import platform
if platform.system() == 'Windows':
    import pyaudiowpatch as pyaudio
else:
    import pyaudio

# from scipy.signal import resample

use_gpu = True

# TODO: handle single-channel & take samplerate from device
# and perhaps convert rate if samplerates differs between input & output devices
channels = 2
samplerate = 48000
# this still works as low as 1024 * 50 but the delay is only marginally lower
# at 1024 * 1023 and samplerate = 48000 the delay is about 1024 * 1023 / 48000 = 21.824 seconds
# at 1024 * 50 it should be ~1sec delay but getting about ~9sec delay

# best results on my gpu takes ~200ms to demix
# so 500ms could be reasonable
buffer_ms = 500
buffer_size = 1024 * 16 #samplerate * buffer_ms // 1000
frames_per_buffer = 1024 * 4 # buffer_size // 4 # divide by sizeof(float32)

input_queue = queue.Queue()
output_queue = queue.Queue()

p = pyaudio.PyAudio()

# dev_in should be CABLE Input
dev_in = None

# dev_out should be Headphones / Speakers
dev_out = None

def print_time(msg, start_time):
    print(msg, '%.2f' % (time.perf_counter() - start_time))

for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    # print(dev)
    if dev['isLoopbackDevice'] and "CABLE Input" in dev['name']:
        dev_in = dev

    if not dev_out and not dev['isLoopbackDevice'] and "Speakers" in dev['name']:
        dev_out = dev

if not dev_in or not dev_out:
    print('CABLE Input or Speakers device not found')
    exit(1)

debug_use_back_buffer = True
volume_vocals = 0.3 # 0.0
def zen_thread(demixer):
    back_buffer = np.zeros((channels, 0), dtype=np.float32)

    assert buffer_size <= demixer.chunk_size, 'buffer_size must be less than or equal to demixer.chunk_size'

    while True:
        input_buffer = input_queue.get()
        # avoid clicking & unnecessary processing if buffer is empty
        if not input_buffer.any():
            print('skipping empty buffer')
            # TODO: check if this may help in order to keep output stream synced
            # output_queue.put(input_buffer)
            continue

        current_buffer_size = input_buffer.shape[1]
        current_buffer_offset = min(back_buffer.shape[1], demixer.chunk_size - current_buffer_size)
        back_buffer = np.concatenate((back_buffer[:, :demixer.chunk_size - current_buffer_size], input_buffer), axis=1)

        print('demixing...')
        stime = time.perf_counter()

        if debug_use_back_buffer:
            output_buffer = demixer.demix(back_buffer, volume_vocals=volume_vocals) #, buffer_size=demixer.chunk_size)
            output_queue.put(output_buffer[:, current_buffer_offset:current_buffer_offset + current_buffer_size])
        else:
            output_buffer = demixer.demix(input_buffer, volume_vocals=volume_vocals)
            output_queue.put(output_buffer)

        print_time('signal demixed:', stime)

        print('back buffer size:', back_buffer.shape[1])

def audio_output_thread(output_stream):
    while True:
        
        if output_queue.empty():
            # TODO: output some default soundwave
            continue

        output_data = output_queue.get()
        output_data_compact = output_data.transpose().flatten() # output_data.reshape((channels, -1)).transpose().flatten()
        print('writing output...')
        stime = time.perf_counter()
        output_stream.write(output_data_compact.tobytes())
        print_time('output written:', stime)


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
            # TODO: raise flag to stop audio processing instantly
            # TODO: try detecting volume from input_data?
            #continue

        input_buffer = np.concatenate((input_buffer, audio_data), axis=1)

        if input_buffer.shape[1] >= buffer_size:
            # print('input buffer filled, processing')
            input_queue.put(input_buffer[:, :buffer_size])
            input_buffer = input_buffer[:, buffer_size:]



def main():
    # NOTE: it may be better loading model before recording starts
    print('loading model...')
    stime = time.perf_counter()
    demixer = zen.ZenDemixer(use_gpu=use_gpu)
    print_time('model loaded:', stime)

    thread = threading.Thread(target=zen_thread, args=(demixer,))
    thread.start()

    try:
        # Open an input stream to capture audio from the loopback device
        # NOTE: the device must not be muted or have volume 0 in the operating system settings
        input_stream = p.open(format=pyaudio.paFloat32,
                              channels=channels,
                              rate=samplerate,
                              input=True,
                              input_device_index=dev_in['index'],
                              frames_per_buffer=frames_per_buffer)

        # Open an output stream to play the processed audio
        output_stream = p.open(format=pyaudio.paFloat32,
                               channels=channels,
                               rate=samplerate,
                               output=True,
                               frames_per_buffer=frames_per_buffer,
                               output_device_index=dev_out['index'])

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
        p.terminate()

if __name__ == "__main__":
    main()
