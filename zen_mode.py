###
### parse args
###

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--list', nargs='?', choices=['input', 'output', 'both'], const='both', help='list audio devices and exit')
parser.add_argument('--virtual-device')
parser.add_argument('--output-device')
parser.add_argument('--model-size', type=int, choices=[5, 6, 7, 8, 9, 10, 11], default=5)
parser.add_argument('--buffer-size', type=int, default=None)
parser.add_argument('--samples-per-io', type=int, default=1024)
parser.add_argument('--volume-multiplier', type=int, default=200, help='audio suffers amplitude loss, once when physical device has <100% volume and again when the virtual playback device has <100% volume. this helps offset it')
parser.add_argument('--volume-vocals', type=int, default=0, help='vocals volume level')
parser.add_argument('--no-gui', action='store_true')
args = parser.parse_args()

###
### initiate startup, install virtual devices if necessary
###

import platform

if platform.system() == 'Windows':
    import pyaudiowpatch as pyaudio
    import device_manager_windows as devman
else:
    import pyaudio
    import device_manager_mac as devman

if args.list:
    pa = pyaudio.PyAudio()

    for idx in range(pa.get_device_count()):
        dev = pa.get_device_info_by_index(idx)
        is_input = dev['maxInputChannels'] > 0
        is_output = dev['maxOutputChannels'] > 0
        if args.list.lower() == 'input' and not is_input:
            continue

        if args.list.lower() == 'output' and not is_output:
            continue

        dev_type = 'both' if is_input and is_output else 'input' if is_input else 'output' if is_output else 'unknown'
        print(f'{dev["name"]} ({dev_type})')

    pa.terminate()
    exit(0)

print('initializing...')

# devman.startup() will install virtual devices if necessary
# running before potentially heavy imports for quick error reporting
try:
    assert devman.startup(), 'ensure virtual devices are installed properly'
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

MODEL_CHANNELS = 2

def print_time(stime, msg, end='\n'):
    print(f'\r{msg}: %.2fs' % (time.perf_counter() - stime), end=end)

class ZenMode:
    pa = None
    demixer = None
    input_stream = None
    output_stream = None
    demixer_thread = None
    playback_thread = None
    record_thread = None

    use_gpu = True
    # possible values between 5 to 11
    model_size = 5
    buffer_size = 1024 * (2 ** (model_size - 1))
    frames_per_buffer = 1024
    volume_music = 1.0
    volume_vocals = 0.0

    dev_in = None
    dev_out = None

    need_restart = False

    def __init__(self):
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

    def load_model(self):
        assert 0 < self.buffer_size < 1024 * (2 ** self.model_size), f'buffer_size must be greater than 0 and less than {1024 * (2 ** self.model_size)}'

        # 128 kB is kind of arbitrary
        assert 0 < self.frames_per_buffer < 1024 * 128, 'samples_per_io must be greater than 0 and less than 1024 * 128'

        self.demixer = zen_demixer.ZenDemixer(use_gpu=self.use_gpu, segment_size=2 ** self.model_size)

    def init_pyaudio(self, dev_in_pattern, dev_out_pattern):
        self.shutdown()

        self.pa = pyaudio.PyAudio()
        self.select_input_device(dev_in_pattern)
        self.select_output_device(dev_out_pattern)

    def select_input_device(self, dev_in_pattern):
        self.dev_in = None
        for idx in range(self.pa.get_device_count()):
            dev = self.pa.get_device_info_by_index(idx)
            if dev['maxInputChannels'] == 0:
                continue

            if re.search(dev_in_pattern, dev['name'], re.IGNORECASE) or dev_in_pattern in dev['name']:
                self.dev_in = dev
                return True

        return False

    def select_output_device(self, dev_out_pattern):
        self.dev_out = None
        for idx in range(self.pa.get_device_count()):
            dev = self.pa.get_device_info_by_index(idx)
            if dev['maxOutputChannels'] == 0:
                continue

            if re.search(dev_out_pattern, dev['name'], re.IGNORECASE) or dev_out_pattern in dev['name']:
                self.dev_out = dev
                return True

        return False

    def demixer_loop(self):
        demixer = self.demixer
        buffer_size = self.buffer_size
        chunk_size = demixer.chunk_size
        assert buffer_size <= chunk_size, 'buffer_size must be less than or equal to demixer.chunk_size'

        back_buffer = np.zeros((MODEL_CHANNELS, 0), dtype=np.float32)

        try:
            while True:
                input_buffer = self.input_queue.get()
                # take the last chunk_size samples
                if len(input_buffer) < chunk_size:
                    remainder = chunk_size - len(input_buffer)
                    input_buffer = np.concatenate((back_buffer[:, -remainder:], input_buffer), axis=1)

                stime = time.perf_counter()
                output_buffer = demixer.demix(input_buffer, volume_music=self.volume_music, volume_vocals=self.volume_vocals) #, buffer_size=demixer.chunk_size)
                print_time(stime, 'demixed audio', end='\r')

                self.output_queue.put(output_buffer[:, -buffer_size:])
                back_buffer = back_buffer[:, -chunk_size:]
        except BaseException as e:
            print(f'demixer thread error')
            print(traceback.format_exc())

    def playback_loop(self):
        try:
            while self.output_stream:
                # if output_queue.empty():
                #     # TODO: output some default soundwave
                #     continue

                output_data = self.output_queue.get()
                output_data_compact = output_data.transpose().flatten() # output_data.reshape((channels, -1)).transpose().flatten()
                if self.volume_music != 1.0:
                    # adjust volume
                    output_data_compact *= self.volume_music

                # writing to output_stream is blocking, so it takes len(output_data_compact) / samplerate seconds to play
                # stime = time.perf_counter()
                self.output_stream.write(output_data_compact.tobytes())
                # print_time(stime, 'played processed audio')
        except BaseException as e:
            print(f'audio output thread error')
            print(traceback.format_exc())

    def record_loop(self):
        prev_empty = None
        input_buffer = np.zeros((MODEL_CHANNELS, 0), dtype=np.float32)
        buffer_size = self.buffer_size

        while self.input_stream:
            if not self.demixer_thread.is_alive():
                print('demixer thread stopped unexpectedly, halting')
                self.shutdown()
                return

            if not self.playback_thread.is_alive():
                print('playback thread stopped unexpectedly, halting')
                self.shutdown()
                return

            # len(input_data) == (frames_per_buffer * channels * sizeof(float))
            input_data = self.input_stream.read(self.frames_per_buffer)

            # len(audio_data) == (frames_per_buffer * channels)
            audio_data_compact = np.frombuffer(input_data, dtype=np.float32)

            # example how to resample, probably not needed
            # demix_samplerate = 44100
            # audio_data_compact = resample(audio_data_compact, int(len(audio_data_compact) * demix_samplerate / samplerate))

            # audio_data_compact has interleaved samples for each channel, convert to separate channels
            audio_data = audio_data_compact.reshape((-1, MODEL_CHANNELS)).transpose()

            if not audio_data.any():
                if not prev_empty:
                    print('empty buffer', end='\r')
                    prev_empty = True
                #continue
            else:
                if prev_empty:
                    print('got audio data', end='\r')
                    prev_empty = False

            input_buffer = np.concatenate((input_buffer, audio_data), axis=1)

            if input_buffer.shape[1] >= buffer_size:
                # print('buffer full, processing')
                self.input_queue.put(input_buffer[:, :buffer_size])
                input_buffer = input_buffer[:, buffer_size:]

    def start(self):
        self.open_input_stream()
        self.open_output_stream()

        if not self.demixer_thread or not self.demixer_thread.is_alive():
            self.demixer_thread = threading.Thread(target=self.demixer_loop, daemon=True)
            self.demixer_thread.start()

        self.playback_thread = threading.Thread(target=self.playback_loop, daemon=True)
        self.playback_thread.start()

        self.record_thread = threading.Thread(target=self.record_loop, daemon=True)
        self.record_thread.start()

    # TODO: figure out what to do if there is only 1 channel
    def open_input_stream(self):
        self.input_stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=MODEL_CHANNELS,
            rate=int(self.dev_in['defaultSampleRate']),
            input=True,
            input_device_index=self.dev_in['index'],
            frames_per_buffer=self.frames_per_buffer
        )

    def open_output_stream(self):
        self.output_stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=MODEL_CHANNELS,
            rate=int(self.dev_in['defaultSampleRate']), # using dev_in here
            output=True,
            frames_per_buffer=self.frames_per_buffer,
            output_device_index=self.dev_out['index']
        )

    def shutdown(self):
        if not self.pa:
            return

        self.input_stream = None
        self.output_stream = None
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join()

        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join()

        self.pa.terminate()
        self.pa = None

def zen_loop(gui_signals, initial_zen_mode_state):
    is_zen_mode = initial_zen_mode_state
    gui_device_name = None

    zen = ZenMode()
    zen.use_gpu = not args.cpu
    zen.model_size = args.model_size
    buffer_base = args.buffer_size if args.buffer_size else (2 ** zen.model_size - 1)
    zen.buffer_size = 1024 * buffer_base
    zen.frames_per_buffer = args.samples_per_io

    zen.volume_music = args.volume_multiplier / 100.0
    zen.volume_vocals = args.volume_vocals / 100.0
    og_volume_multiplier = zen.volume_music

    dev_in_pattern = args.virtual_device
    if not dev_in_pattern:
        dev_in_pattern = 'BlackHole|CABLE Input'

    dev_out_pattern = args.output_device
    if not dev_out_pattern:
        dev_out_pattern = 'Speakers|Headphones'

    try:
        zen.init_pyaudio(dev_in_pattern, dev_out_pattern)

        assert zen.dev_in, f'input device not found: {dev_in_pattern}'
        assert zen.dev_out, f'output device not found: {dev_out_pattern}'

        print('-' * 80)
        print(f'input device: {zen.dev_in["index"]} {zen.dev_in["name"]} ({zen.dev_in["maxInputChannels"]} channels, samplerate {zen.dev_in["defaultSampleRate"]})')
        print(f'output device: {zen.dev_out["index"]} {zen.dev_out["name"]} ({zen.dev_out["maxOutputChannels"]} channels, samplerate {zen.dev_out["defaultSampleRate"]})')
        print(f'preferred processor: {"cpu" if args.cpu else "gpu"}')
        print(f'buffer size: {zen.buffer_size}')
        print(f'samples per io: {zen.frames_per_buffer}')
        print('=' * 80)

        print("[starting audio processing. press CTRL+C to stop.]")

        print('loading model...', end='\r')
        stime = time.perf_counter()
        zen.load_model()
        print_time(stime, 'model loaded')

        if is_zen_mode:
            zen.start()
            devman.set_virtual_audio_device_as_default()

        # set volume of both devices to 33%
        devman.set_volume(zen.dev_in['name'], 0.33)
        devman.set_volume(zen.dev_out['name'], 0.33)

        gui_device_name = zen.dev_out['name']

        # on windows it needs a little bit of time or it crashes?!
        # (windows support is hell, api is so buggy)
        time.sleep(2)

        prev_volume = None
        while True: # zen.record_thread.is_alive():
            try:
                if 'device' in gui_signals:
                    gui_device_name = gui_signals.pop('device')
                    # reuse zen_mode signal because it is easier...
                    gui_signals['zen_mode'] = is_zen_mode

                if 'zen_mode' in gui_signals:
                    is_zen_mode = gui_signals.pop('zen_mode')
                    if is_zen_mode:
                        # in zen mode need to refresh the input device... so init and start()
                        zen.init_pyaudio(dev_in_pattern, gui_device_name)
                        zen.start()

                        # again, sleep because of stupid windows
                        time.sleep(2)
                        devman.set_virtual_audio_device_as_default()
                    else:
                        zen.shutdown()
                        devman.set_default_output_device(gui_device_name)

                if 'volume_vocals' in gui_signals:
                    zen.volume_vocals = gui_signals.pop('volume_vocals')

                if 'volume_music' in gui_signals:
                    zen.volume_music = gui_signals.pop('volume_music') * og_volume_multiplier
                    # devman.set_volume(zen.dev_in['name'], gui_signals['volume_music'])

                virtual_device_volume = devman.get_system_volume()
                if virtual_device_volume != prev_volume:
                    prev_volume = virtual_device_volume
                    # NOTE: do not change volume_music since both the playback & the virtual device volumes are affected
                    # zen.volume_music = virtual_device_volume * og_volume_multiplier
                    # print('setting virtual device volume to', int(100*virtual_device_volume)
                    devman.set_volume(zen.dev_out['name'], virtual_device_volume)
            except BaseException as e:
                # print(f'error getting system volume: {e}')
                pass

            time.sleep(1)
    except KeyboardInterrupt:
        pass
    except BaseException as e:
        print(f"an error occurred")
        print(traceback.format_exc())
    finally:
        print("\nshutting down...")
        zen.shutdown()
        exit(1)


def main():
    # this must be called before initializing PyAudio
    atexit.register(devman.restore_default_audio_device)

    if args.no_gui:
        zen_loop({}, True)
        return

    gui_signals = {}
    zen_thread = threading.Thread(target=zen_loop, args=(gui_signals, False), daemon=True)
    zen_thread.start()

    # gui moved to main thread because otherwise it won't exit properly on finish...
    import zen_gui
    gui = zen_gui.ZenGui()

    def on_zen_mode_change(value):
        gui_signals['zen_mode'] = value

    def on_volume_change(type, value):
        gui_signals['volume_' + type] = value

    def on_device_select(name):
        gui_signals['device'] = name

    gui.on_zen_mode_change = on_zen_mode_change
    gui.on_volume_change = on_volume_change
    gui.on_device_select = on_device_select

    gui.root.mainloop()

    # TODO: this is just a temporary solution. need to hide window instead of exit and have tray icon or something
    exit(0)

main()
