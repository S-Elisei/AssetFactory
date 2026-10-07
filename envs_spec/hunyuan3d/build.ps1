# Usage: build.ps1. Run after install.ps1 hunyuan3d. Checks out Hunyuan3D-2 at its pinned commit into
# envs\hunyuan3d\src and makes the hy3dgen package importable. Idempotent.
# It also builds the custom_rasterizer (CUDA) and mesh_processor (C++) extensions of hy3dgen.texgen with MSVC 2022 and
# CUDA 12.6 when they are not built yet. hunyuan3d.patch is applied to the checkout.
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "..\build_tools.ps1")
$Root = Split-Path (Split-Path $PSScriptRoot)
$Venv = Join-Path $Root "envs\hunyuan3d"
$Py = Join-Path $Venv "Scripts\python.exe"
$Src = Join-Path $Venv "src\Hunyuan3D-2"

Sync-PinnedRepo "https://github.com/Tencent/Hunyuan3D-2.git" "f8db63096c8282cb27354314d896feba5ba6ff8a" $Src
Invoke-Checked { git -C $Src checkout -q --force HEAD }
Invoke-Checked { git -C $Src apply --whitespace=nowarn (Join-Path $PSScriptRoot "hunyuan3d.patch") }

$Renderer = Join-Path $Src "hy3dgen\texgen\differentiable_renderer"
$HasRasterizer = Test-Python $Py "import torch, custom_rasterizer"
$HasProcessor = [bool](Get-ChildItem $Renderer -Filter "mesh_processor*.pyd")
if (-not ($HasRasterizer -and $HasProcessor)) {
    Install-BuildTools $Py
    Enter-CudaBuildEnv
}
if (-not $HasRasterizer) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $Src "hy3dgen\texgen\custom_rasterizer\build")
    Invoke-Checked { uv pip install --python $Py --no-build-isolation --no-cache --reinstall-package custom_rasterizer (Join-Path $Src "hy3dgen\texgen\custom_rasterizer") }
}
if (-not $HasProcessor) {
    Push-Location $Renderer
    try { Invoke-Checked { & $Py setup.py build_ext --inplace } } finally { Pop-Location }
}

Add-SourcePath $Py "af_hy3dgen_src" @($Src)
Invoke-Checked { & $Py -c "from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline" }
Invoke-Checked { & $Py -c "import torch, custom_rasterizer; from hy3dgen.texgen.differentiable_renderer import mesh_processor; assert mesh_processor.__file__.endswith('.pyd')" }
