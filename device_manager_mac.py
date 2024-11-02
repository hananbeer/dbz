import subprocess

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
    return sas_run('-a')

def sas_set_audio_device_default(device_id):
    return sas_run('-s', device_id)

def sas_set_audio_device_mute_state(device_id, is_mute):
    return sas_run('-m', 'mute' if is_mute else 'unmute', device_id)

# def sas_audio_device_change_volume(device_id, volume_delta):
#     return sas_run('...')

def get_audio_devices():
    return sas_list_audio_devices()

def init_macos(try_install=True):
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

        return init_macos(False)

    # TODO: what to do if there are multuple devices of this name?
    blackhole_devices = blackhole_devices[0]
    print('blackhole device id:', blackhole_devices)

    # TODO: if device is not active, it will not be listed by svv.
    # need to test further for this edge case
    # if vb_cable_device['state'] != 'Active':
    #     print('VB-Cable installed but not active, enabling device...')
    #     svv_enable_audio_device(vb_cable_device['id'])
    #     return False

    # if vb_cable_device['is_muted']:
    #     print('VB-Cable device is muted, unmuting...')
    #     svv_set_audio_device_mute_state(vb_cable_device['id'], False)

    # # no need to check volume, just always add 100 to it
    # svv_audio_device_change_volume(vb_cable_device['id'], 100)

    return True

init_macos()
exit(0)
