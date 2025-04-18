@echo off
CLS

:: Check for administrator privileges
NET SESSION >nul 2>&1
if %errorLevel% == 0 (
    echo Administrator privileges detected.
    :: Change directory to the location of the batch file after gaining admin rights
    cd /d "%~dp0"
) else (
    echo Requesting administrator privileges...
    goto :runasadmin
)

:: Your original batch file commands go below this line
:: Activate the virtual environment
call .venv\Scripts\activate.bat

:: Start the TTS trainer
python train_tts_model.py --base-dir voice_datasets --character Zhongli --output-dir tts_output

:: Pause to keep the terminal open after execution
pause

exit /b %errorLevel%

:: --- Do not edit below this line ---
:runasadmin
echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
echo UAC.ShellExecute "%~s0", "", "", "runas", 1 >> "%temp%\getadmin.vbs"
"%temp%\getadmin.vbs"
exit /b %errorLevel%
