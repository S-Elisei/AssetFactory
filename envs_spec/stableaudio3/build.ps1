# Usage: build.ps1. Run after install.ps1 stableaudio3. Checks out stable-audio-3 at its pinned commit into
# envs\stableaudio3\src\stable-audio-3 and makes the stable_audio_3 package importable. Idempotent.
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "..\build_tools.ps1")
$Root = Split-Path (Split-Path $PSScriptRoot)
$Venv = Join-Path $Root "envs\stableaudio3"
$Py = Join-Path $Venv "Scripts\python.exe"
$Src = Join-Path $Venv "src\stable-audio-3"

Sync-PinnedRepo "https://github.com/Stability-AI/stable-audio-3.git" "779434a908193105335fd8d833418603625b2859" $Src

Add-SourcePath $Py "af_stableaudio_src" @($Src)
Invoke-Checked { & $Py -c "from stable_audio_3 import StableAudioModel" }
