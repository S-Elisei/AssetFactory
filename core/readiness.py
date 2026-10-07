"""Readiness of local stages and cloud apps, read from the markers that `core.install` writes. A marker is an empty
file:
- env marker `<root>/envs/<env>/.installed`: written after the install and the build of the environment;
- weights marker `<MODELS>/.installed/<env>/<stage>`: written after the download mode of the Worker returned for the
  stage module, whether or not the module has `download()`;
- app marker `<MODELS>/.installed/cloud/<app>`: written after the weights download and the deploy of the Modal app."""
from core.layout import ENVS, MODELS, ROOT

INSTALL = r"envs\core\Scripts\python.exe -m core.install"


def env_marker(env):
    return ENVS / env / ".installed"


def weights_marker(env, stage):
    return MODELS / ".installed" / env / stage


def app_marker(app):
    return MODELS / ".installed" / "cloud" / app


def stage_names(env):
    """The stage modules of `<root>/stages/<env>/`, sorted."""
    return sorted(path.stem for path in (ROOT / "stages" / env).glob("*.py"))


def stage_readiness(env, stage):
    """None when the local stage `stage` of the environment `env` is ready (the env marker and the weights marker
    exist); else the command, run from the repository root, that makes it ready."""
    if env_marker(env).exists() and weights_marker(env, stage).exists():
        return None
    return f"{INSTALL} {env}"


def app_readiness(app):
    """None when the Modal app `app` (the module name in `cloud/`) is ready (its marker exists); else the command, run
    from the repository root, that makes it ready."""
    if app_marker(app).exists():
        return None
    return f"{INSTALL} {app}"
