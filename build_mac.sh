pyinstaller zen_mode.py -y --exclude-module pkg_resources --icon=img/zen-256.ico --hidden-import pyaudio --collect-submodules pyaudio
pyinstaller zen_mode.spec -y
mkdir -p dist/img
cp img/zen-256.png dist/img/.
mkdir -p dist/models
cp models/UVR-MDX-NET-Inst_HQ_3.onnx dist/models/.
