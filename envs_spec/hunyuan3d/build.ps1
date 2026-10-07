# Usage: build.ps1. Run after install.ps1 hunyuan3d. Checks out Hunyuan3D-2 at its pinned commit into
# envs\hunyuan3d\src and makes the hy3dgen package importable. Idempotent.
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "..\build_tools.ps1")
$Root = Split-Path (Split-Path $PSScriptRoot)
$Venv = Join-Path $Root "envs\hunyuan3d"
$Py = Join-Path $Venv "Scripts\python.exe"
$Src = Join-Path $Venv "src\Hunyuan3D-2"

Sync-PinnedRepo "https://github.com/Tencent/Hunyuan3D-2.git" "f8db63096c8282cb27354314d896feba5ba6ff8a" $Src

Add-SourcePath $Py "af_hy3dgen_src" @($Src)
Invoke-Checked { & $Py -c "from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline" }
