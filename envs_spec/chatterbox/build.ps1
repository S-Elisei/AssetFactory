# Usage: build.ps1. Run after install.ps1 chatterbox. Installs chatterbox-tts from GitHub at its pinned commit without
# its dependencies. An installed package from the same archive is kept. Idempotent.
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "..\build_tools.ps1")
$Root = Split-Path (Split-Path $PSScriptRoot)
$Py = Join-Path $Root "envs\chatterbox\Scripts\python.exe"
$Commit = "5de7a54aa4e5e2baadb0182dde554908b48b85c2"

Invoke-Checked { uv pip install --python $Py --no-deps "chatterbox-tts @ https://github.com/resemble-ai/chatterbox/archive/$Commit.zip" }
Invoke-Checked { & $Py -c "import perth; from chatterbox.mtl_tts import ChatterboxMultilingualTTS" }
