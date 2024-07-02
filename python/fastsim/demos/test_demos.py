import subprocess
import os
from pathlib import Path
from fastsim import fastsimrust
import pytest


def demo_paths():
    demo_paths = list(Path(__file__).parent.glob("*demo*.py"))
    demo_paths.remove(Path(__file__).resolve())
    return demo_paths

REQUIRED_FEATURES = {"vehicle_import_demo": "vehicle-import"}

@pytest.mark.parametrize(
    "demo_path", demo_paths(), ids=[dp.name for dp in demo_paths()]
)
def test_demo(demo_path: Path):
    if demo_path.stem in REQUIRED_FEATURES.keys() and REQUIRED_FEATURES[demo_path.stem] not in fastsimrust.enabled_features():
        pytest.skip(f'requires "{REQUIRED_FEATURES[demo_path.stem]}" feature')

    os.environ["SHOW_PLOTS"] = "false"
    os.environ["TESTING"] = "true"

    rslt = subprocess.run(
        ["python", demo_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert rslt.returncode == 0, rslt.stderr
