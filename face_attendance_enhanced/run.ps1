<#
 Sets up (first run only) and starts the app.
 Run from the project folder:
     powershell -ExecutionPolicy Bypass -File .\run.ps1
 (-ExecutionPolicy Bypass applies to this one command only; nothing is changed permanently.)
#>
param([switch]$Test, [switch]$InitOnly)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment (Python 3.11)..."
    if (Get-Command py -ErrorAction SilentlyContinue) { py -3.11 -m venv .venv } else { python -m venv .venv }
    & .\.venv\Scripts\python.exe -m pip install --upgrade pip
    & .\.venv\Scripts\python.exe -m pip install -r requirements.txt
}

# The venv's python is called directly, so no Activate.ps1 (and no execution-policy change) is needed.
& .\.venv\Scripts\python.exe -c "from database import Database; from config import Config; Database(Config.DATABASE_PATH).init_db(); print('Database ready:', Config.DATABASE_PATH)"

if ($Test)     { & .\.venv\Scripts\python.exe -m pytest -q; exit $LASTEXITCODE }
if ($InitOnly) { exit 0 }

Write-Host "Starting http://127.0.0.1:5000  (Ctrl+C to stop)"
Start-Process "http://127.0.0.1:5000"
& .\.venv\Scripts\python.exe app.py
