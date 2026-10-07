"""Paths of the factory. Importing this module puts `<root>/worker` on sys.path, which makes the module `context` of
`worker/context.py` (MODELS, InputError) importable."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENVS = ROOT / "envs"

sys.path.insert(0, str(ROOT / "worker"))
from context import MODELS  # noqa: E402
