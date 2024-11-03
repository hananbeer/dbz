import subprocess

default_audio_devices = []

def install_vb_cable():
    # as shell so it will request elevated permissions if needed
    return subprocess.run(r'.\bin\VBCABLE_Setup_x64.exe -i -h', shell=True)

def svv_run(*args):
    return subprocess.run(['./bin/SoundVolumeView.exe', *args])

def svv_dump_audio_devices(output_path):
    return svv_run('/scomma', output_path, '/Columns', 'Type,Name,Item ID,Direction,Device State,Default,Muted')

def svv_enable_audio_device(device_id):
    return svv_run('/Enable', device_id)

def svv_set_audio_device_default(device_id):
    return svv_run('/SetDefault', device_id, 'all')

def svv_set_audio_device_mute_state(device_id, is_mute):
    return svv_run('/Mute' if is_mute else '/Unmute', device_id)

def svv_audio_device_change_volume(device_id, volume_delta):
    return svv_run('/ChangeVolume', device_id, str(volume_delta))

def get_audio_devices():
    svv_dump_audio_devices('dev_list.csv')

    f = open('dev_list.csv', 'r')
    lines = f.read().split('\n')[1:-1]
    devices = []
    for line in lines:
        dev_type, name, item_id, direction, state, default, muted = line.split(',')
        devices.append({ 'id': item_id, 'name': name, 'direction': direction, 'state': state, 'type': dev_type, 'default': default, 'is_muted': muted == 'True' })

    return devices

def restore_default_audio_device():
    if default_audio_devices:
        print('restoring default audio device', default_audio_devices)
        svv_set_audio_device_default(default_audio_devices[0]['id'])

def startup(try_install=True):
    all_devices = get_audio_devices()

    # filter out Application-specific devices and only get Render devices
    devices = [dev for dev in all_devices if dev['type'] == 'Device' and dev['direction'] == 'Render']

    # find VB Cable input device
    # "CABLE Input" ID: {0.0.0.00000000}.{c2a849eb-7157-4779-a1f6-e0518a26ef8e}
    vb_cable_devices = [dev for dev in devices if dev['name'] == 'CABLE Input']
    if not vb_cable_devices:
        if not try_install:
            print('VB-Cable device not found, installation failed')
            return False

        print('VB-Cable device not found, installing...')

        # install vb cable driver (requires admin)
        install_vb_cable()

        print('VB-Cable installer finished')

        return startup(False)

    # TODO: what to do if there are multuple devices of this name?
    vb_cable_device = vb_cable_devices[0]
    # print('vb device id:', vb_cable_device)

    # TODO: if device is not active, it will not be listed by svv.
    # need to test further for this edge case
    # if vb_cable_device['state'] != 'Active':
    #     print('VB-Cable installed but not active, enabling device...')
    #     svv_enable_audio_device(vb_cable_device['id'])
    #     return False

    if vb_cable_device['is_muted']:
        print('VB-Cable device is muted, unmuting...')
        svv_set_audio_device_mute_state(vb_cable_device['id'], False)

    # no need to check volume, just always add 100 to it
    svv_audio_device_change_volume(vb_cable_device['id'], 100)

    if vb_cable_device['default'] != 'Render':
        global default_audio_devices
        default_audio_devices = [dev for dev in devices if dev['default'] == 'Render']

        # TODO: set default to render
        print('VB-Cable device is not default output, setting to default...')
        svv_set_audio_device_default(vb_cable_device['id'])

    return True
