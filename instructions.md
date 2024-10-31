for model to run in mac libsndfile needs to be installed like so:
conda install -c conda-forge libsndfile

to record audio output, need to:
(sudo) brew install blackhole-2ch
brew install portaudio
brew link portaudio (may not be necessary?)
pip install pyaudio


notice I needed to go to settings -> privacy & security -> microphone and add "cursor" to the allowed apps list.
(need to figure out how to allow python/my specific program)

brew install switchaudio-osx
SwitchAudioSource -a
SwitchAudioSource -s "BlackHole 2ch"
SwitchAudioSource -s "MacBook Pro Speakers"


# NOTE

(at least on mac, possibly windows too)
the recorded volume depends on the virtual device volume
but also the playing volume depends on the speakers volume
but also the volume controls only control the default device (which should be the virtual device)
and it seems at least on mac the virtual device INPUT volume controls the recording volume, but the keyboard shortcuts control only the virtual OUTPUT volume
so there's a lot to unpack here

(probably keep volume at 100% for the virtual devices)

to set the default device to the virtual device:
SwitchAudioSource -s "MacBook Pro Speakers"
osascript -e "set volume output volume 100 --100%"
SwitchAudioSource -s "BlackHole 2ch"
osascript -e "set volume input volume 100 --100%"
