# Usage: build.ps1. Run after install.ps1 sf3d. Checks out stable-fast-3d at its pinned commit into envs\sf3d\src with
# sf3d.patch applied, builds its texture_baker (CUDA) and uv_unwrapper (C++) extensions with MSVC 2022 and CUDA 12.6 when they are not built,
# and makes the sf3d package importable. Idempotent.
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "..\build_tools.ps1")
$Root = Split-Path (Split-Path $PSScriptRoot)
$Venv = Join-Path $Root "envs\sf3d"
$Py = Join-Path $Venv "Scripts\python.exe"
$Src = Join-Path $Venv "src\stable-fast-3d"

Sync-PinnedRepo "https://github.com/Stability-AI/stable-fast-3d.git" "ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2" $Src
Invoke-Checked { git -C $Src checkout -q --force HEAD }
Invoke-Checked { git -C $Src apply --whitespace=nowarn (Join-Path $PSScriptRoot "sf3d.patch") }

$Missing = @("texture_baker", "uv_unwrapper") | Where-Object { -not (Test-Python $Py "import torch, $_") }
if ($Missing) {
    Install-BuildTools $Py
    Enter-CudaBuildEnv
    $env:USE_CUDA = "1"
    $env:USE_NATIVE_ARCH = "0"
}
foreach ($Name in $Missing) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $Src "$Name\build")
    Invoke-Checked { uv pip install --python $Py --no-build-isolation --no-cache --reinstall-package $Name (Join-Path $Src $Name) }
}

Add-SourcePath $Py "af_sf3d_src" @($Src)
Invoke-Checked { & $Py -c "import texture_baker, uv_unwrapper, sf3d.system" }
