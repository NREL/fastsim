"""Test suite for FASTSim demo scripts to ensure they run without errors."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


# Per-demo wall-clock limit
DEMO_TIMEOUT_SECONDS = 600


def demo_paths():
    """Get list of all demo script paths."""
    return list(Path(__file__).parent.rglob("demo*.py"))


@pytest.mark.parametrize("demo_path", demo_paths(), ids=[dp.name for dp in demo_paths()])
def test_demo(demo_path: Path):
    """Test that each demo script runs successfully without errors."""
    os.environ["SHOW_PLOTS"] = "false"
    os.environ["PYTEST"] = "true"
    try:
        rslt = subprocess.run(
            [sys.executable, demo_path],
            capture_output=True,
            text=True,
            timeout=DEMO_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{demo_path.name} did not finish within {DEMO_TIMEOUT_SECONDS} s")

    assert rslt.returncode == 0, rslt.stderr
