pyinstaller zen_mode.py -y --exclude-module pkg_resources --icon=img/zen-256.ico --windowed
pyinstaller zen_mode.spec -y
cp img/zen-256.png dist/zen_mode.app/Contents/Resources/img/zen-256.png
cp models/UVR-MDX-NET-Inst_HQ_3.onnx dist/zen_mode.app/Contents/Resources/models/UVR-MDX-NET-Inst_HQ_3.onnx
