# Usage: build.ps1. Run after install.ps1 qwentts. Installs qwen-tts from GitHub at its pinned commit without its
# dependencies. An installed package from the same archive is kept. Idempotent.
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "..\build_tools.ps1")
$Root = Split-Path (Split-Path $PSScriptRoot)
$Py = Join-Path $Root "envs\qwentts\Scripts\python.exe"
$Commit = "022e286b98fbec7e1e916cb940cdf532cd9f488e"

Invoke-Checked { uv pip install --python $Py --no-deps "qwen-tts @ https://github.com/QwenLM/Qwen3-TTS/archive/$Commit.zip" }
Invoke-Checked { & $Py -c "from qwen_tts import Qwen3TTSModel" }
