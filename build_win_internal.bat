@echo off

set PROVIDER=%1
set VERSION=%2
if "%VERSION%" == "" (
    for /f "usebackq tokens=*" %%a in (`git rev-parse HEAD`) do set VERSION=%%a
)
echo version: %VERSION:~0,8%

echo Installing requirements
pip install -r requirements-%PROVIDER%.txt || goto error
pip install pyinstaller || goto error
echo Building spec file for %PROVIDER%
set TARGET_DIR=dist-%VERSION%-win-%PROVIDER%
pyinstaller zen_mode.py -y --icon=img/zen-256.ico --windowed --exclude-module torch_directml --distpath=%TARGET_DIR% || goto error
echo Building executable for %PROVIDER%
pyinstaller zen_mode.spec -y || goto error
echo Copying executable to %TARGET_DIR%
xcopy /s /i /y bin\* %TARGET_DIR%\zen_mode\bin\.
xcopy /s /i /y img\zen-256.png %TARGET_DIR%\zen_mode\img\zen-256.png
xcopy /s /i /y models\UVR-MDX-NET-Inst_HQ_3.onnx %TARGET_DIR%\zen_mode\models\UVR-MDX-NET-Inst_HQ_3.onnx

goto done

:error
echo Error!

:done
