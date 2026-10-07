# Usage: install.ps1 <env>. Creates envs\<env> (Python 3.12) if absent and installs into it
# envs_spec\<env>\requirements-torch.txt, when that file exists, then envs_spec\<env>\requirements.txt.
param([Parameter(Mandatory)][string]$Name)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "build_tools.ps1")
$Root = Split-Path $PSScriptRoot
$Venv = Join-Path $Root (Join-Path "envs" $Name)
$Py = Join-Path $Venv "Scripts\python.exe"
$Spec = Join-Path $PSScriptRoot $Name

if (-not (Test-Path $Py)) {
    uv venv $Venv --python 3.12
    if ($LASTEXITCODE -ne 0) { throw "uv venv failed with exit code $LASTEXITCODE" }
}
$Files = @("requirements.txt")
if (Test-Path (Join-Path $Spec "requirements-torch.txt")) { $Files = @("requirements-torch.txt") + $Files }
foreach ($File in $Files) {
    uv pip install --python $Py -r (Join-Path $Spec $File)
    if ($LASTEXITCODE -ne 0) { throw "uv pip install failed with exit code $LASTEXITCODE" }
}
