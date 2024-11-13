@echo off

if "%CONDA_DEFAULT_ENV:~-12%" == "dbz-directml" (
    set EXTRA_FLAGS=--hidden-import torch_directml
    call build_win_internal.bat directml %1
    xcopy /s /i /y /d %CONDA_PREFIX%\Lib\site-packages\torch_directml %TARGET_DIR%\zen_mode\_internal\torch_directml\.
    echo Done
) else (
    echo Incorrect conda environment! First create it:
    echo     conda create -n dbz-directml python==3.10.15
    echo Then to activate it:
    echo     conda activate dbz-directml
)

@REM TODO: zip %TARGET_DIR%
