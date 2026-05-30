@echo off
setlocal
cd /d %~dp0

if not exist .venv\Scripts\python.exe (
  echo ???? install.bat
  exit /b 1
)

for %%f in (wheels\legal_pii_redactor-*.whl) do set WHEEL=%%f
if not defined WHEEL (
  echo ??? wheel ??
  exit /b 1
)

.venv\Scripts\python.exe -m pip install --force-reinstall "%WHEEL%"
