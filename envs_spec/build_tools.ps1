# Helpers dot-sourced by the build.ps1 files of the environments that check out upstream code or compile torch
# extensions, and by install.ps1. Dot-sourcing sets the cache folder of uv for the script.

$env:UV_CACHE_DIR = Join-Path (Split-Path $PSScriptRoot) ".cache\uv"

function Invoke-Checked {
    # Runs a native command given as a script block and throws when it exits non-zero.
    param([scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "command failed with exit code ${LASTEXITCODE}: $Command" }
}

function Test-Python {
    # True when $Python runs $Code without error.
    param([string]$Python, [string]$Code)
    & $Python -c $Code 2>$null | Out-Null
    return $LASTEXITCODE -eq 0
}

function Sync-PinnedRepo {
    # Makes $Dir a shallow checkout of $Url at exactly $Commit.
    param([string]$Url, [string]$Commit, [string]$Dir)
    if ((Test-Path (Join-Path $Dir ".git")) -and ((git -C $Dir rev-parse HEAD) -eq $Commit)) { return }
    if (Test-Path $Dir) { Remove-Item -Recurse -Force $Dir }
    New-Item -ItemType Directory -Force $Dir | Out-Null
    Invoke-Checked { git -C $Dir init -q }
    Invoke-Checked { git -C $Dir fetch -q --depth 1 $Url $Commit }
    Invoke-Checked { git -C $Dir checkout -q --force FETCH_HEAD }
}

function Install-BuildTools {
    # Installs into the environment the packages that the extension builds need besides torch.
    param([string]$Python)
    Invoke-Checked { uv pip install --python $Python setuptools==80.9.0 wheel==0.45.1 ninja==1.11.1.4 }
}

function Enter-CudaBuildEnv {
    # Imports the MSVC 2022 x64 developer environment and points torch's extension builder at the CUDA 12.6 toolkit
    # named by CUDA_PATH_V12_6 (set by the toolkit installer).
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    $vs = & $vswhere -version "[17.0,18.0)" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
    $vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
    $env:PATH = "$(Split-Path $vswhere);$env:PATH"
    cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
        if ($_ -match "^([^=]+)=(.*)$") { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] }
    }
    $cuda = $env:CUDA_PATH_V12_6
    $env:CUDA_HOME = $cuda
    $env:CUDA_PATH = $cuda
    $env:PATH = "$cuda\bin;$env:PATH"
    $env:DISTUTILS_USE_SDK = "1"
    $env:TORCH_CUDA_ARCH_LIST = "8.9"
}

function Add-SourcePath {
    # Writes site-packages\<Name>.pth listing $Paths, which makes the code in those folders importable.
    param([string]$Python, [string]$Name, [string[]]$Paths)
    $site = (& $Python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])").Trim()
    [IO.File]::WriteAllText((Join-Path $site "$Name.pth"), (($Paths -join "`n") + "`n"))
}
