r"""Core code of the factory. It runs in the `core` environment, which is installed once by
`pwsh -NoProfile -ExecutionPolicy Bypass -File envs_spec\install.ps1 core`; every module runs as
`envs\core\Scripts\python.exe -m core.<module>`, and this command and every install command run from the repository
root."""
