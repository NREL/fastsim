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
    # TODO: finish populating this with all of the python-exposed SerdeAPI
    # methods and `to_dataframe`
    def to_pydict(self, data_fmt: str = "msg_pack", flatten: bool = False) -> Dict: ...
    @classmethod
    def from_pydict(
        cls, pydict: Dict, data_fmt: str = "msg_pack", skip_init: bool = False
    ) -> Self: ...

class SimDrive(SerdeAPI): ...  # TODO: flesh out more
