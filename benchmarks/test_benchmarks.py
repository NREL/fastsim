import subprocess
import os
from pathlib import Path
import pytest
import sys

PYTHON_EXE = Path(sys.executable)

def script_paths():
    script_paths = list(Path(__file__).parent.glob("*.py"))
    script_paths.remove(Path(__file__).resolve())
    # can't test f2.py because it needs `fastsim-2` environment
    script_paths.remove(Path(__file__).parent / "./f2.py")
    return script_paths

@pytest.mark.parametrize(
    "script_path", script_paths(), ids=[sp.name for sp in script_paths()])
def test_demo(script_path: Path):
    os.environ['SHOW_PLOTS'] = "false"
    rslt = subprocess.run(
        [str(PYTHON_EXE), script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    assert rslt.returncode == 0, rslt.stderr
