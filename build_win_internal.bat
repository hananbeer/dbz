@echo off

set PROVIDER=%1
set VERSION=%2
if "%VERSION%" == "" set VERSION=no-ver

echo Installing requirements
pip install -r requirements-%PROVIDER%.txt || goto error
pip install pyinstaller || goto error
echo Building spec file for %PROVIDER%
pyinstaller zen_mode.py -y --icon=img/zen-256.ico --windowed --exclude-module torch_directml || goto error
echo Building executable for %PROVIDER%
pyinstaller zen_mode.spec -y || goto error
set TARGET_DIR=dist-%VERSION%-win-%PROVIDER%
echo Copying executable to %TARGET_DIR%
copy bin %TARGET_DIR%\zen_mode\bin
copy img\zen-256.png %TARGET_DIR%\zen_mode\img
copy models\UVR-MDX-NET-Inst_HQ_3.onnx %TARGET_DIR%\zen_mode\models\UVR-MDX-NET-Inst_HQ_3.onnx

goto done

:error
echo Error!

:done
