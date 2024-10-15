from __future__ import annotations
from typing_extensions import Self
from typing import Dict, List, Tuple, Optional, ByteString
from abc import ABC
from fastsim.vehicle import VEHICLE_DIR
import yaml
from pathlib import Path

class SerdeAPI(object):
    def init(self): ...
    def from_file(file_path: Path) -> Self: ...
    # TODO: finish populating this