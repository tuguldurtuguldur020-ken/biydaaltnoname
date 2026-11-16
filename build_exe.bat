@echo off
setlocal
cd /d "%~dp0"
call run.bat --install
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install "pyinstaller>=6.11,<7"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m tools.build_exe
if errorlevel 1 goto failed
echo Ready: dist\PlateWatch\PlateWatch.exe
pause
exit /b 0
:failed
echo Build failed. Check the error above.
pause
exit /b 1
