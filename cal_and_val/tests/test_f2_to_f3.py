"""Copies vehicle files in f3 format to f3 format"""

from pathlib import Path

import fastsim as fsim


def test_f2_to_f3() -> None:
    """Copies vehicle files in f3 format to f3 format"""
    source_dir = Path(__file__).parents[1] / "f2-vehicles"
    assert source_dir.exists()
    target_dir = Path(__file__).parents[1] / "f3-vehicles"
    assert target_dir.exists()

    for f2file in source_dir.iterdir():
        if f2file.suffix != ".yaml":
            continue
        f3veh = fsim.Vehicle.from_f2_file(f2file)
        f3veh.to_file(target_dir / f2file.name)
