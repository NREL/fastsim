"""Test suite for FASTSim demo scripts to ensure they run without errors."""
import os
import subprocess
import sys
from pathlib import Path

import pytest


def demo_paths():
    """Get list of all demo script paths."""
    demo_paths = list(Path(__file__).parent.glob("*demo*.py"))
    demo_paths.remove(Path(__file__).resolve())
    return demo_paths


@pytest.mark.parametrize("demo_path", demo_paths(), ids=[dp.name for dp in demo_paths()])
def test_demo(demo_path: Path):
    """Test that each demo script runs successfully without errors."""
    os.environ["SHOW_PLOTS"] = "false"
    rslt = subprocess.run(
        [sys.executable, demo_path],
        capture_output=True,
        text=True,
    )

    assert rslt.returncode == 0, rslt.stderr
