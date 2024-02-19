import subprocess
import os
from pathlib import Path
import pytest


ADDITIONAL_ARGS = {"calibration_demo": ["-p", "1"]}


def demo_paths():
    demo_paths = list(Path(__file__).parent.glob("*demo*.py"))
    demo_paths.remove(Path(__file__).resolve())
    return demo_paths


@pytest.mark.parametrize(
    "demo_path", demo_paths(), ids=[dp.name for dp in demo_paths()]
)
def test_demo(demo_path: Path):
    os.environ["SHOW_PLOTS"] = "false"
    rslt = subprocess.run(
        (
            ["python", demo_path]
            + (
                ADDITIONAL_ARGS[demo_path.stem]
                if demo_path.stem in ADDITIONAL_ARGS.keys()
                else []
            )
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert rslt.returncode == 0, rslt.stderr
