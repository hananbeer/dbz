import subprocess

default_audio_device = None

def install_blackhole():
    return subprocess.run(['brew', 'install', 'blackhole-2ch'])

def install_switchaudio():
    return subprocess.run(['brew', 'install', 'switchaudio-osx'])

def osa_run(*args):
    return subprocess.run(['osascript', *args])

def sas_run(*args):
    return subprocess.check_output(['SwitchAudioSource', *args]).decode('utf-8')

def sas_list_audio_devices():
    return sas_run('-a').strip().split('\n')

def sas_get_default_audio_device():
    return sas_run('-c').strip()

def sas_set_default_audio_device(device_id):
    return sas_run('-s', device_id)

def sas_set_audio_device_mute_state(device_id, is_mute):
    return sas_run('-m', 'mute' if is_mute else 'unmute', device_id)

# def sas_audio_device_change_volume(device_id, volume_delta):
#     return sas_run('...')

def get_audio_devices():
    return sas_list_audio_devices()

def restore_default_audio_device():
    if default_audio_device:
        sas_set_default_audio_device(default_audio_device)

def startup(try_install=True):
    all_devices = get_audio_devices()

    # find VB Cable input device
    # "CABLE Input" ID: {0.0.0.00000000}.{c2a849eb-7157-4779-a1f6-e0518a26ef8e}
    blackhole_devices = [dev for dev in all_devices if 'blackhole' in dev.lower()]
    if not blackhole_devices:
        if not try_install:
            print('blackhole device not found, installation failed')
            return False

        print('blackhole device not found, installing...')

        # install vb cable driver (requires password)
        install_blackhole()

        print('blackhole installer finished')

        return startup(False)

    # TODO: what to do if there are multuple devices of this name?
    blackhole_device = blackhole_devices[0]
    # print('blackhole device id:', blackhole_device)

    # TODO: check if active/disabled, muted, volume
    
    global default_audio_device
    default_audio_device = sas_get_default_audio_device()
    sas_set_default_audio_device(blackhole_device)

    return True
