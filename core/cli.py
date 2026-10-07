r"""Command line of the factory: `envs\core\Scripts\python.exe -m core.cli <command>`, run from the repository root.
Every command is a subparser of `build_parser` whose `run` default is a function of the parsed arguments; it returns
nothing, and an exception of `subprocess.run(check=True)` ends `main` with the exit code of the failed step.

    install <target>   target: a local environment, a Modal app, or `all` (every local environment, and every Modal
                       app that is not ready). An environment runs its install and build only when its env marker is
                       absent, and downloads only the stages whose weights markers are absent. A Modal app named
                       explicitly is deployed again."""
import argparse
import subprocess
import sys

from core.install import cloud_apps, install_all, install_app, install_env, local_envs


def run_install(args):
    if args.target == "all":
        install_all()
    elif args.target in local_envs():
        install_env(args.target)
    else:
        install_app(args.target)


def build_parser():
    parser = argparse.ArgumentParser(prog="core.cli", description="AssetFactory command line.")
    commands = parser.add_subparsers(dest="command", required=True)
    install = commands.add_parser("install", help="install environments and weights, deploy Modal apps")
    install.add_argument("target", choices=["all", *local_envs(), *cloud_apps()])
    install.set_defaults(run=run_install)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.run(args)
    except subprocess.CalledProcessError as error:
        print(f"stopped: {subprocess.list2cmdline(error.cmd)} exited with code {error.returncode}", file=sys.stderr)
        return error.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
