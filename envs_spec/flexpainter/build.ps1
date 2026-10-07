# Usage: build.ps1. Run after install.ps1 flexpainter. Checks out FlexPainter, torchsparse (with torchsparse.patch applied)
# and sparsehash-c11 at their pinned commits into envs\flexpainter\src\FlexPainter, envs\flexpainter\src\torchsparse and
# envs\flexpainter\src\sparsehash-c11; a checkout already at its commit is used as it is. Builds torchsparse, torch-scatter
# and nvdiffrast with MSVC 2022 and CUDA 12.6 (an installed torch-scatter and nvdiffrast are kept), installs the flash_attn
# shim (flash_attn.py) into the environment and makes the FlexPainter packages importable. Idempotent.
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "..\build_tools.ps1")
$Root = Split-Path (Split-Path $PSScriptRoot)
$Venv = Join-Path $Root "envs\flexpainter"
$Py = Join-Path $Venv "Scripts\python.exe"
$FlexPainter = Join-Path $Venv "src\FlexPainter"
$TorchSparse = Join-Path $Venv "src\torchsparse"
$SparseHash = Join-Path $Venv "src\sparsehash-c11"

Sync-PinnedRepo "https://github.com/StarRealMan/FlexPainter.git" "14ad9167f601a3a56a2474c8fe3aa438d4554441" $FlexPainter
Sync-PinnedRepo "https://github.com/mit-han-lab/torchsparse.git" "385f5ce8718fcae93540511b7f5832f4e71fd835" $TorchSparse
Sync-PinnedRepo "https://github.com/sparsehash/sparsehash-c11.git" "0c748d9528c06f8360b0edf6766d0f64a5f48034" $SparseHash
Invoke-Checked { git -C $TorchSparse checkout -q --force HEAD }
Invoke-Checked { git -C $TorchSparse apply --whitespace=nowarn (Join-Path $PSScriptRoot "torchsparse.patch") }

# One-line headers google/dense_hash_map, dense_hash_set, sparse_hash_map and sparse_hash_set redirect to sparsehash-c11.
$Shim = Join-Path $SparseHash "shim\google"
New-Item -ItemType Directory -Force $Shim | Out-Null
foreach ($Name in "dense_hash_map", "dense_hash_set", "sparse_hash_map", "sparse_hash_set") {
    [IO.File]::WriteAllText((Join-Path $Shim $Name), "#include <sparsehash/$Name>`n")
}

Install-BuildTools $Py
Enter-CudaBuildEnv
$env:FORCE_CUDA = "1"
$env:MAX_JOBS = $env:NUMBER_OF_PROCESSORS
$env:INCLUDE = "$(Split-Path $Shim);$SparseHash;$env:INCLUDE"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $TorchSparse "build")
Invoke-Checked { uv pip install --python $Py --no-build-isolation --no-cache --reinstall-package torchsparse $TorchSparse }
Invoke-Checked { uv pip install --python $Py --no-build-isolation --no-cache torch-scatter==2.1.2 "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae" }

$Site = (& $Py -c "import sysconfig; print(sysconfig.get_paths()['purelib'])").Trim()
Copy-Item -Force (Join-Path $PSScriptRoot "flash_attn.py") $Site

Add-SourcePath $Py "af_flexpainter_src" @($FlexPainter)
Invoke-Checked { & $Py -c "import torch_scatter, torchsparse, nvdiffrast.torch, spconv.pytorch, flash_attn, open_clip, Imath, OpenEXR; from pipeline.outpainter import OutpainterPipe; from spuv.mesh_utils import vertex_transform" }
