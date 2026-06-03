# build.ps1 — Build the DocProcessorPro one-directory distributable
#
# Prerequisites (run once):
#   pip install pyinstaller
#
# Usage:
#   .\build.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $Python)) {
    Write-Host 'ERROR: .venv not found. Run "python -m venv .venv" and install dependencies first.' -ForegroundColor Red
    exit 1
}

Write-Host "Cleaning previous build artifacts..." -ForegroundColor Cyan
foreach ($dir in @("build", "dist")) {
    $path = Join-Path $ProjectRoot $dir
    if (Test-Path $path) {
        Remove-Item $path -Recurse -Force
        Write-Host "  Removed $dir/"
    }
}

Write-Host "Running PyInstaller..." -ForegroundColor Cyan
Set-Location $ProjectRoot
& $Python -m PyInstaller DocProcessorPro.spec --noconfirm

if ($LASTEXITCODE -ne 0) {
    Write-Host "PyInstaller failed (exit code $LASTEXITCODE)." -ForegroundColor Red
    exit $LASTEXITCODE
}

$DistDir = Join-Path $ProjectRoot 'dist\DocProcessorPro'
$ExePath = Join-Path $DistDir 'DocProcessorPro.exe'
Write-Host ''
Write-Host 'Build complete.' -ForegroundColor Green
Write-Host ('Output: ' + $DistDir)
Write-Host ''
Write-Host 'Quick smoke-test - launch the app:'
Write-Host ('  ' + $ExePath)
