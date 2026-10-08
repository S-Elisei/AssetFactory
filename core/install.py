r"""Install of local environments and of Modal apps: `envs\core\Scripts\python.exe -m core.install <target>`, run from
the repository root. `target` is a local environment, a Modal app, or `all` (every local environment, and every Modal
app that is not ready). An environment runs its install and build only when its env marker is absent, and downloads
only the stages whose weights markers are absent; a Modal app named explicitly is deployed again. A step that fails
stops the install with its exit code and leaves no marker for it; the markers are those of `core.readiness`. Output of
every step goes to the console."""
import argparse
import importlib
import subprocess
import sys

import modal

from core.layout import ENVS, MODELS, ROOT
from core.readiness import app_marker, app_readiness, env_marker, stage_names, weights_marker

SPEC = ROOT / "envs_spec"
POWERSHELL = ["pwsh", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]


def local_envs():
    """The environments with a requirements file in `envs_spec/`, except `core`."""
    return sorted(path.name for path in SPEC.iterdir() if path.name != "core" and (path / "requirements.txt").exists())


def cloud_apps():
    """The Modal apps: the modules of `cloud/`."""
    return sorted(path.stem for path in (ROOT / "cloud").glob("*.py") if path.stem != "__init__")


def _run(command):
    subprocess.run(command, cwd=ROOT, check=True)


def _mark(marker):
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


def install_env(env):
    """When the env marker is absent: runs `install.ps1 <env>`, then `envs_spec/<env>/build.ps1` when it exists, then
    writes the env marker. Then runs the download mode of the Worker for every stage module of `stages/<env>/` that has
    no weights marker, writing the marker after each download."""
    marker = env_marker(env)
    if not marker.exists():
        _run([*POWERSHELL, str(SPEC / "install.ps1"), env])
        build = SPEC / env / "build.ps1"
        if build.exists():
            _run([*POWERSHELL, str(build)])
        _mark(marker)
    python = ENVS / env / "Scripts" / "python.exe"
    for stage in stage_names(env):
        marker = weights_marker(env, stage)
        if not marker.exists():
            _run([str(python), str(ROOT / "worker" / "main.py"), env, "download", stage])
            _mark(marker)


def install_app(app):
    """Fills the Volume of the Modal app with `download_weights` in an ephemeral run of `weights_app` of the module,
    then deploys `app` of the module, then calls `ready` of its class, whose container takes the memory snapshot, then
    writes the app marker. The token in the file `<MODELS>/hf/token` is passed
    to `download_weights`."""
    module = importlib.import_module(f"cloud.{app}")
    token = MODELS / "hf" / "token"
    if not token.exists():
        raise SystemExit(f"{token} is absent: write the Hugging Face token into it, then install {app} again.")
    marker = app_marker(app)
    marker.unlink(missing_ok=True)
    with modal.enable_output():
        with module.weights_app.run():
            module.download_weights.remote(token.read_text(encoding="utf-8").strip())
        module.app.deploy()
        modal.Cls.from_name(module.APP_NAME, module.CLASS)().ready.remote()
    _mark(marker)


def install_all():
    """Installs every local environment, and every Modal app that is not ready."""
    for env in local_envs():
        install_env(env)
    for app in cloud_apps():
        if app_readiness(app) is not None:
            install_app(app)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="core.install", description="Installs environments and Modal apps.")
    parser.add_argument("target", choices=["all", *local_envs(), *cloud_apps()])
    target = parser.parse_args().target
    try:
        if target == "all":
            install_all()
        elif target in local_envs():
            install_env(target)
        else:
            install_app(target)
    except subprocess.CalledProcessError as error:
        print(f"stopped: {subprocess.list2cmdline(error.cmd)} exited with code {error.returncode}", file=sys.stderr)
        sys.exit(error.returncode)
