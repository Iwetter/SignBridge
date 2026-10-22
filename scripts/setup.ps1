param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Use a Windows CPython 3.11 or 3.12 installation with venv support.' }
}
& ./.venv/Scripts/python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& ./.venv/Scripts/python.exe scripts/download_model.py
if ($LASTEXITCODE -ne 0) { throw 'Landmarker model download failed.' }
& ./.venv/Scripts/python.exe -m pip freeze | Set-Content -LiteralPath requirements-lock.txt
Write-Host 'Ready: ./.venv/Scripts/python.exe app.py'
