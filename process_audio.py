import pyaudiowpatch as pyaudio
import numpy as np

frames_per_buffer = 1024

class AudioProcessor:
    def __init__(self):
        self.buffer_ms = 1500
        self.samplerate = 48000
        self.buffer_size = int(self.buffer_ms * self.samplerate / 1024.)

    def set_input_device(self, device_index):
        self.input_device_index = device_index

    def set_output_device(self, device_index):
        self.output_device_index = device_index

    def read_samples(self, sample_count):
        pass

channels = 2
samplerate = 48000

"""
Device 21: CABLE Input (VB-Audio Virtual Cable) [Loopback]
  Input channels: 2
  Output channels: 0
"""
input_device_index = 21

"""
Device 5: Speakers (Realtek(R) Audio)
  Input channels: 0
  Output channels: 2
"""
output_device_index = 5

def main():
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

        # Main processing loop
        while True:
            # len(input_data) == (frames_per_buffer * channels * sizeof(float))
            input_data = input_stream.read(frames_per_buffer)
            output_stream.write(input_data)
            continue
            
            # len(audio_data) == (frames_per_buffer * channels)
            audio_data = np.frombuffer(input_data, dtype=np.float32)

            if not audio_data.any():
                print('empty buffer')
                continue
            
            # Process the audio data (example: increase volume)
            processed_audio = [audio_data[i] * ((i % 10) / 10) for i in range(len(audio_data))]
            
            # Ensure the processed audio is within the valid range (-1 to 1)
            processed_audio = np.clip(processed_audio, -1, 1)
            
            # Convert the processed audio back to bytes
            output_data = processed_audio.tobytes()
            
            # Write the processed audio to the output stream
            output_stream.write(output_data)

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
