# Usage: build.ps1. Run after install.ps1 acestep. Checks out ACE-Step 1.5 at its pinned commit into
# envs\acestep\src\ACE-Step-1.5 and makes the acestep package importable. Idempotent.
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "..\build_tools.ps1")
$Root = Split-Path (Split-Path $PSScriptRoot)
$Venv = Join-Path $Root "envs\acestep"
$Py = Join-Path $Venv "Scripts\python.exe"
$Src = Join-Path $Venv "src\ACE-Step-1.5"

Sync-PinnedRepo "https://github.com/ace-step/ACE-Step-1.5.git" "ca1e85fe9430179831e6bc6be790c332190a3866" $Src

Add-SourcePath $Py "af_acestep_src" @($Src)
Invoke-Checked { & $Py -c "from acestep.handler import AceStepHandler; from acestep.llm_inference import LLMHandler; from acestep.inference import generate_music; from acestep.models.turbo.modeling_acestep_v15_turbo import AceStepConditionGenerationModel" }
