$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot "venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Virtual environment not found at '$python'. Create it with: python -m venv venv"
}

Set-Location $projectRoot
& $python -m uvicorn src.api.main:app --reload --port 8000
