@echo off
setlocal
cd /d %~dp0

if not exist .venv\Scripts\python.exe (
  echo ???? install.bat
  exit /b 1
)

set LEGAL_PII_USE_NER=1
.venv\Scripts\python.exe app.py
