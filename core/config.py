"""The settings of the factory: `<root>/config.yaml`, read once on import."""
import yaml

from core.layout import ROOT

SETTINGS = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
