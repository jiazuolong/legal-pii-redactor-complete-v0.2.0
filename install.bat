@echo off
setlocal
cd /d %~dp0

if not exist .venv\Scripts\python.exe (
  py -3.10 -m venv .venv
  if errorlevel 1 goto :error
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto :error

pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo ?????
echo ?? Web ?: start_web.bat
echo ?? CLI ?: install_library.bat
exit /b 0

:error
echo.
echo ???????? Python 3.10+ ??????
exit /b 1
