# Creates envs\<folder name of this script> (Python 3.12) and installs requirements.txt into it.
$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot)
$env:UV_CACHE_DIR = Join-Path $Root ".cache\uv"
$Venv = Join-Path $Root (Join-Path "envs" (Split-Path $PSScriptRoot -Leaf))
$Py = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path $Py)) {
    uv venv $Venv --python 3.12
    if ($LASTEXITCODE -ne 0) { throw "uv venv failed with exit code $LASTEXITCODE" }
}
uv pip install --python $Py -r (Join-Path $PSScriptRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "uv pip install failed with exit code $LASTEXITCODE" }
