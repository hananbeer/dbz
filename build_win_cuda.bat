@echo off

if "%CONDA_DEFAULT_ENV:~-8%" == "dbz-cuda" (
    call build_win_internal.bat cuda %1
    echo Done
) else (
    echo Incorrect conda environment! First create it:
    echo     conda create -n dbz-cuda python==3.10.15
    echo Then to activate it:
    echo     conda activate dbz-cuda
)
