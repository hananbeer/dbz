@echo off

if "%CONDA_DEFAULT_ENV:~-12%" == "dbz-directml" (
    call build_win_internal.bat directml %1
    copy %CONDA_PREFIX%\Lib\site-packages\torch_directml dist-directml\zen_mode\.
    echo Done
) else (
    echo Incorrect conda environment! First create it:
    echo     conda create -n dbz-directml python==3.10.15
    echo Then to activate it:
    echo     conda activate dbz-directml
)
