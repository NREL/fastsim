import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union
import shutil
import datetime

import fastsim as fsim


def print_dt():
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def copy_demo_files(path_for_copies: Path = Path("demos")):
    """
    Copies demo files from demos folder into specified local directory
    # Arguments
    - `path_for_copies`: path to copy files into (relative or absolute in)
    # Warning
    Running this function will overwrite existing files with the same name in the specified directory, so
    make sure any files with changes you'd like to keep are renamed.
    """
    v = f"v{fsim.__version__}"
    current_demo_path = fsim.package_root() / "demos"
    assert Path(path_for_copies).resolve() != Path(current_demo_path), (
        "Can't copy demos inside site-packages"
    )
    demo_files = list(current_demo_path.glob("*demo*.py"))
    test_demo_files = list(current_demo_path.glob("*test*.py"))
    for file in test_demo_files:
        demo_files.remove(file)
    for file in demo_files:
        if os.path.exists(path_for_copies):
            dest_file = Path(path_for_copies) / file.name
            shutil.copy(file, path_for_copies)
            with open(dest_file, "r+") as file:
                file_content = file.readlines()
                prepend_str = f"# %% Copied from FASTSim version '{v}'. Guaranteed compatibility with this version only.\n"
                prepend = [prepend_str]
                file_content = prepend + file_content
                file.seek(0)
                file.writelines(file_content)
            print(f"Saved {dest_file.name} to {dest_file}")
        else:
            os.makedirs(path_for_copies)
            dest_file = Path(path_for_copies) / file.name
            shutil.copy(file, path_for_copies)
            with open(dest_file, "r+") as file:
                file_content = file.readlines()
                prepend_str = f"# %% Copied from FASTSim version '{v}'. Guaranteed compatibility with this version only.\n"
                prepend = [prepend_str]
                file_content = prepend + file_content
                file.seek(0)
                file.writelines(file_content)
            print(f"Saved {dest_file.name} to {dest_file}")
