from pathlib import Path

from . import fastsimrust
from .fastsimrust import *

def package_root() -> Path:
    """Returns the package root directory."""
    return Path(__file__).parent
