import pyaudiowpatch as pyaudio
import numpy as np
import zen

frames_per_buffer = 1024
channels = 2
samplerate = 48000
chunk_size = 3 * samplerate # 1024 * 1023

# volume between 0..1
volume = 0.5

input_device_index = 21
output_device_index = 5

def main():
    print('initializing')
    demixer = zen.ZenDemixer()

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

        input_buffer = np.zeros((channels, 0), dtype=np.float32)
        output_buffer = np.zeros((channels, 0), dtype=np.float32)

        # Main processing loop
        while True:
            # len(input_data) == (frames_per_buffer * channels * sizeof(float))
            input_data = input_stream.read(frames_per_buffer)

            # len(audio_data) == (frames_per_buffer * channels)
            audio_data_compact = np.array(np.frombuffer(input_data, dtype=np.float32))
            # audio_data_compact has interleaved samples for each channel, convert to separate channels
            audio_data = audio_data_compact.reshape((-1, channels)).transpose()
            # audio_data = np.array([audio_data_compact[::2], audio_data_compact[1::2]])

            input_buffer = np.concatenate((input_buffer, audio_data), axis=1)

            # if not audio_data.any():
            #     print('empty buffer')
            #     #continue

            # TODO: pad buffer and process
            if input_buffer.shape[1] >= 3 * chunk_size:
                print('demixing')
                output_data = demixer.demix(input_buffer[:, :chunk_size])
                print('writing output')
                # output_data_compact = np.array([], dtype=np.float32)
                # for i in range(0, output_data.shape[1]):
                #     output_data_compact = np.append(output_data_compact, output_data[0, i])
                #     output_data_compact = np.append(output_data_compact, output_data[1, i])

                output_data_compact = output_data.reshape((channels, -1)).transpose().flatten()
                output_stream.write(output_data_compact.tobytes())
                print('output written')
                input_buffer = input_buffer[:, chunk_size:]

            # TODO: process buffer
            # Process the audio data (example: increase volume)
            # audio_data = [audio_data[i] * ((i % 10) / 10) for i in range(len(audio_data))]
            
            # Ensure the processed audio is within the valid range (-1 to 1)
            # audio_data = np.clip(audio_data, -1, 1)
            # Convert the processed audio back to bytes
            #output_data = (buffer[:, :frames_per_buffer] * volume)
            
            # if output_buffer:
            #     # Write the processed audio to the output stream
            #     output_data = output_buffer
            #     output_stream.write(output_data.tobytes())


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
