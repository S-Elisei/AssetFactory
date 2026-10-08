"""Paths of the factory. Importing this module puts `<root>/worker` on sys.path, which makes the module `context` of
`worker/context.py` (MODELS, InputError) importable, and appends `<root>/shared`, the modules the input checks of the
cloud stages use."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENVS = ROOT / "envs"
DATA = ROOT / "data"

sys.path.insert(0, str(ROOT / "worker"))
sys.path.append(str(ROOT / "shared"))
from context import MODELS  # noqa: E402
