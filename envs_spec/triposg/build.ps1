# Usage: build.ps1. Run after install.ps1 triposg. Checks out TripoSG at its pinned commit into envs\triposg\src with
# triposg.patch applied, builds the diso (DiffDMC) CUDA extension with MSVC 2022 and CUDA 12.6 when it does not import,
# and makes the triposg
# package importable. Idempotent.
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "..\build_tools.ps1")
$Root = Split-Path (Split-Path $PSScriptRoot)
$Venv = Join-Path $Root "envs\triposg"
$Py = Join-Path $Venv "Scripts\python.exe"
$Src = Join-Path $Venv "src\TripoSG"
$MvAdapter = Join-Path $Venv "src\MV-Adapter"

Sync-PinnedRepo "https://github.com/VAST-AI-Research/TripoSG.git" "fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c" $Src
Invoke-Checked { git -C $Src checkout -q --force HEAD }
Invoke-Checked { git -C $Src apply --whitespace=nowarn (Join-Path $PSScriptRoot "triposg.patch") }
# It also checks out MV-Adapter at its pinned commit into envs\triposg\src\MV-Adapter with mv-adapter.patch applied and
# installs nvdiffrast at its pinned commit, built with the same toolchain. A checkout already at the commit is used as it
# is; an installed nvdiffrast at the commit is kept.
Sync-PinnedRepo "https://github.com/huanngzh/MV-Adapter.git" "4277e0018232bac82bb2c103caf0893cedb711be" $MvAdapter
Invoke-Checked { git -C $MvAdapter checkout -q --force HEAD }
Invoke-Checked { git -C $MvAdapter apply --whitespace=nowarn (Join-Path $PSScriptRoot "mv-adapter.patch") }

$HasDiso = Test-Python $Py "import torch, diso"
$HasNvdiffrast = Test-Python $Py "import torch, nvdiffrast.torch"
if (-not ($HasDiso -and $HasNvdiffrast)) {
    Install-BuildTools $Py
    Enter-CudaBuildEnv
}
if (-not $HasDiso) { Invoke-Checked { uv pip install --python $Py --no-build-isolation --no-cache diso==0.1.4 } }
if (-not $HasNvdiffrast) {
    Invoke-Checked { uv pip install --python $Py --no-build-isolation --no-cache "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae" }
}

Add-SourcePath $Py "af_triposg_src" @($Src)
Invoke-Checked { & $Py -c "import diso; from triposg.pipelines.pipeline_triposg import TripoSGPipeline" }
Add-SourcePath $Py "af_mvadapter_src" @($MvAdapter)
Invoke-Checked { & $Py -c "import nvdiffrast.torch; from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline; from mvadapter.utils.mesh_utils import CameraProjection; from scripts.inference_ig2mv_sdxl import preprocess_image" }
