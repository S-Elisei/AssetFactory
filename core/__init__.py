r"""Core code of the factory. It runs in the `core` environment, which is installed once by
`pwsh -NoProfile -ExecutionPolicy Bypass -File envs_spec\install.ps1 core`. The factory is started by
`envs\core\Scripts\python.exe -m core`; the other modules run as `envs\core\Scripts\python.exe -m core.<module>`. This
command and every install command run from the repository root."""
