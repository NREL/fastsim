import os
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union
import shutil
import datetime

import fastsim as fsim


def print_dt():
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def set_log_level(level: Union[str, int]) -> int:
    """
    Sets logging level for both Python and Rust FASTSim.
    The default logging level is WARNING (30).
    https://docs.python.org/3/library/logging.html#logging-levels

    Parameters
    ----------
    level: `str` | `int`
        Logging level to set. `str` level name or `int` logging level

        =========== ================
        Level       Numeric value
        =========== ================
        CRITICAL    50
        ERROR       40
        WARNING     30
        INFO        20
        DEBUG       10
        NOTSET      0

    Returns
    -------
    `int`
        Previous log level
    """
    # Map string name to logging level

    allowed_args = [
        ("CRITICAL", 50),
        ("ERROR", 40),
        ("WARNING", 30),
        ("INFO", 20),
        ("DEBUG", 10),
        ("NOTSET", 0),
        # no logging of anything ever!
        ("NONE", logging.CRITICAL + 1),
    ]
    allowed_str_args = [a[0] for a in allowed_args]
    allowed_int_args = [a[1] for a in allowed_args]

    err_str = f"Invalid arg: '{level}'.  See doc string:\n{set_log_level.__doc__}"

    if isinstance(level, str):
        assert level.upper() in allowed_str_args, err_str
        level = logging._nameToLevel[level.upper()]
    else:
        assert level in allowed_int_args, err_str

    # Extract previous log level and set new log level
    fastsim_logger = logging.getLogger("fastsim")
    previous_level = fastsim_logger.level
    fastsim_logger.setLevel(level)
    fastsimrust_logger = logging.getLogger("fastsim_core")
    fastsimrust_logger.setLevel(level)
    return previous_level


def disable_logging() -> int:
    """
    Disable FASTSim logs from being shown by setting log level
    to CRITICAL+1 (51).

    Returns
    -------
    `int`
        Previous log level
    """
    return set_log_level(logging.CRITICAL + 1)


def enable_logging(level: Optional[Union[int, str]] = None) -> int:
    """
    Re-enable FASTSim logging, optionally to a specified log level,
    otherwise to the default WARNING (30) level.

    Parameters
    ----------
    level: `str` | `int`, optional
        Logging level to set. `str` level name or `int` logging level.
        See `utils.set_log_level()` docstring for more details on logging levels.

    Returns
    -------
    `int`
        Previous log level
    """
    if level is None:
        level = logging.WARNING
    return set_log_level(level)


@contextmanager
def with_logging(log_level="DEBUG"):
    """
    Enable then disable logging using a context manager.
    Logging is re-enabled even if the nested code throws an error.

    # Arguments
    - `log_level`: see levels in `set_log_level`

    # Example:
    ``` python
    with fastsim.utils.suppress_logging():
        ...  # do stuff with logging suppressed
    ```
    """
    previous_level = enable_logging(log_level)

    try:
        yield
    finally:
        set_log_level(previous_level)


@contextmanager
def without_logging():
    """
    Disable, then re-enable FASTSim logging using a context manager.
    The log level is returned to its previous value.
    Logging is re-enabled even if the nested code throws an error.

    Example:
    ``` python
    with fastsim.utils.without_logging():
        ...  # do stuff with logging suppressed
    ```
    """
    previous_level = disable_logging()
    try:
        yield
    finally:
        enable_logging(previous_level)


def set_log_filename(filename: Union[str, Path]):
    handler = logging.FileHandler(filename)
    handler.setFormatter(logging.root.handlers[0].formatter)
    logging.getLogger("fastsim").addHandler(handler)
    logging.getLogger("fastsim.fastsimrust").addHandler(handler)


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
