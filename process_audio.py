import zen
import time
import queue
import threading
import numpy as np
import pyaudiowpatch as pyaudio

# from scipy.signal import resample

use_gpu = True

frames_per_buffer = 4096
channels = 2
samplerate = 48000
# this still works as low as 1024 * 50 but the delay is only marginally lower
# at 1024 * 1023 and samplerate = 48000 the delay is about 1024 * 1023 / 48000 = 21.824 seconds
# at 1024 * 50 it should be ~1sec delay but getting about ~9sec delay
buffer_size = 1024 * 1023

# volume between 0..1
#volume = 0.5

input_device_index = 21
output_device_index = 5

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

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    try:
        # Get the default WASAPI loopback device
        # loopback_device = p.get_default_wasapi_loopback()

        # Open an input stream to capture audio from the loopback device
        # NOTE: the device must not be muted or have volume 0 in the operating system settings
        input_stream = p.open(format=pyaudio.paFloat32,
                              channels=channels,
                              rate=samplerate,
                              input=True,
                              input_device_index=input_device_index,#loopback_device['index'],
                              frames_per_buffer=frames_per_buffer)

        # Open an output stream to play the processed audio
        output_stream = p.open(format=pyaudio.paFloat32,
                               channels=channels,
                               rate=samplerate,
                               output=True,
                               frames_per_buffer=frames_per_buffer,
                               output_device_index=output_device_index)

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
        p.terminate()

if __name__ == "__main__":
    main()
