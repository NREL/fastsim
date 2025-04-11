import sys
import subprocess
import os
from pathlib import Path
import pytest


def demo_paths():
    demo_paths = list(Path(__file__).parent.glob("*demo*.py"))
    demo_paths.remove(Path(__file__).resolve())
    return demo_paths


@pytest.mark.parametrize("demo_path", demo_paths(), ids=[dp.name for dp in demo_paths()])
def test_demo(demo_path: Path):
    os.environ["SHOW_PLOTS"] = "false"
    rslt = subprocess.run(
        [sys.executable, demo_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    assert rslt.returncode == 0, rslt.stderr
