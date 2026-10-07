# Usage: build.ps1. Run after install.ps1 triposg. Checks out TripoSG at its pinned commit into envs\triposg\src with
# triposg.patch applied, builds the diso (DiffDMC) CUDA extension with MSVC 2022 and CUDA 12.6, and makes the triposg
# package importable. Idempotent.
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "..\build_tools.ps1")
$Root = Split-Path (Split-Path $PSScriptRoot)
$env:UV_CACHE_DIR = Join-Path $Root ".cache\uv"
$Venv = Join-Path $Root "envs\triposg"
$Py = Join-Path $Venv "Scripts\python.exe"
$Src = Join-Path $Venv "src\TripoSG"

Sync-PinnedRepo "https://github.com/VAST-AI-Research/TripoSG.git" "fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c" $Src
Invoke-Checked { git -C $Src checkout -q --force HEAD }
Invoke-Checked { git -C $Src apply --whitespace=nowarn (Join-Path $PSScriptRoot "triposg.patch") }

Install-BuildTools $Py
Enter-CudaBuildEnv
Invoke-Checked { uv pip install --python $Py --no-build-isolation --no-cache diso==0.1.4 }

Add-SourcePath $Py "af_triposg_src" @($Src)
Invoke-Checked { & $Py -c "import diso; from triposg.pipelines.pipeline_triposg import TripoSGPipeline" }
