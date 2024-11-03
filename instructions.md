# How It Works

The original audio playback needs to be recorded, processed with AI and finally replaced with AI-processed audio.

Normally an audio device is either an input (microphone) or an output device (speakers), so a virtual audio device is needed that is both an input and an output device.

The virtual audio devices are installed via VB-Cable on Windows and BlackHole on MacOS.

Once a virtual audio device is installed, it needs to be set as the default output device, and it is used as an input device for the program.

The processed audio is sent to the physical output device directly.

Note this makes it possible to change specific applications audio output in the system mixer settings - using the virtual device for AI processing or the physical device to bypass the program.

## TODO:

- copy volume levels when changed
- monitor audio devices plugged/unplugged
- reduce build size (avoiding torch is probably the largest size reduction, but onnx could result in slower runtime)
- add a tray icon
- add splash
- ui
- keyboard shortcuts

# Windows

`SoundVolumeView.exe` (by same developer as `NirCmd.exe`, unfortunately neither open source but incredibly useful)

```bat
rem get list of devices
SoundVolumeView.exe /scomma list.csv /Columns "Name,Command-Line Friendly ID"
rem set default output device
SoundVolumeView.exe /SetDefault "VB-Cable Input" all
SoundVolumeView.exe /SetDefault Speakers all
SoundVolumeView.exe /SetDefault Headphones all
```

# MacOS

to record audio output, need to:
```sh
# (sudo?)
brew install blackhole-2ch
brew install portaudio
brew link portaudio (may not be necessary?)
pip install pyaudio
```


notice I needed to go to `Settings -> Privacy & Security -> Microphone` and add "cursor" (yes the IDE wtf?) to the allowed apps list.
(need to figure out how to allow python/my specific program)

```sh
brew install switchaudio-osx
# list audio devices
SwitchAudioSource -a
# set default output device
SwitchAudioSource -s "BlackHole 2ch"
SwitchAudioSource -s "MacBook Pro Speakers"
```


# NOTE

(at least on mac, possibly windows too)
the recorded volume depends on the virtual device volume
but also the playing volume depends on the speakers volume
but also the volume controls only control the default device (which should be the virtual device)
and it seems at least on mac the virtual device INPUT volume controls the recording volume, but the keyboard shortcuts control only the virtual OUTPUT volume
so there's a lot to unpack here

(probably keep volume at 100% for the virtual devices)

to set the default device to the virtual device:
```sh
SwitchAudioSource -s "MacBook Pro Speakers"
osascript -e "set volume output volume 100 --100%"
SwitchAudioSource -s "BlackHole 2ch"
osascript -e "set volume input volume 100 --100%"
```


for zen_from_file.py to run in mac libsndfile needs to be installed like so:
```sh
conda install -c conda-forge libsndfile
```

# Building

```sh
conda create -n dbz python=3.10.15
conda activate dbz
conda create -n dbz python=3.10.15
conda activate dbz
pip install -r requirements.txt

# on windows:
pip install pyinstaller==6.11.0
pyinstaller process_audio.py -y --onefile
pyinstaller process_audio.spec -y

# on macos:
conda install -c conda-forge pyinstaller
pyinstaller --exclude-module pkg_resources process_audio.py -y --onefile
pyinstaller process_audio.spec -y

```
