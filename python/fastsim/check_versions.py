#!/usr/bin/env python3

"""
This script reads the FASTSim project version from `pyproject.toml` and compares it
to the versions declared in the Cargo.toml files for the workspace packages, ensuring
that all versions are consistent.
It also verifies that the `fastsim-proc-macros` dependency in
`fastsim-core/Cargo.toml` has the correct path and version.
"""

import json
import pathlib
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:
    print("error: python3 with tomllib support (3.11+) is required", file=sys.stderr)
    sys.exit(2)


def load_pyproject_version(pyproject_path: pathlib.Path) -> str:
    pyproject = load_toml(pyproject_path)
    return pyproject["project"]["version"]


def load_toml(path: pathlib.Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_workspace_packages(repo_root: pathlib.Path) -> tuple[pathlib.Path, list[tuple[str, str, str]]]:
    result = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    workspace_root = pathlib.Path(metadata["workspace_root"]).resolve()

    package_versions = []
    for package in metadata["packages"]:
        manifest_path = pathlib.Path(package["manifest_path"]).resolve()
        try:
            relative_manifest = manifest_path.relative_to(workspace_root)
        except ValueError:
            relative_manifest = manifest_path

        package_versions.append((str(relative_manifest), package["name"], package["version"]))

    return workspace_root, sorted(package_versions)


def check_fastsim_proc_macros_dependency(repo_root: pathlib.Path, expected_version: str) -> list[str]:
    manifest_path = repo_root / "fastsim-core" / "Cargo.toml"
    manifest = load_toml(manifest_path)
    dependencies = manifest.get("dependencies", {})
    proc_macros_dependency = dependencies.get("fastsim-proc-macros")

    if not isinstance(proc_macros_dependency, dict):
        return [
            "fastsim-core/Cargo.toml: dependency 'fastsim-proc-macros' must be declared as an inline table"
        ]

    problems = []
    actual_path = proc_macros_dependency.get("path")
    actual_version = proc_macros_dependency.get("version")

    if actual_path != "fastsim-proc-macros":
        problems.append(
            "fastsim-core/Cargo.toml: dependency 'fastsim-proc-macros' must set path = \"fastsim-proc-macros\""
        )

    if actual_version != expected_version:
        problems.append(
            "fastsim-core/Cargo.toml: dependency 'fastsim-proc-macros' must set "
            f"version = \"{expected_version}\""
        )

    return problems


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    pyproject_path = repo_root / "pyproject.toml"
    expected_version = load_pyproject_version(pyproject_path)
    _workspace_root, package_versions = load_workspace_packages(repo_root)

    problems = []
    for manifest_path, package_name, cargo_version in package_versions:
        if cargo_version != expected_version:
            problems.append(
                f"{manifest_path}: package {package_name!r} has version {cargo_version}, expected {expected_version}"
            )

    problems.extend(check_fastsim_proc_macros_dependency(repo_root, expected_version))

    if problems:
        print("Version mismatch detected:", file=sys.stderr)
        print(f"  pyproject.toml: project.version = {expected_version}", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print(f"Versions match: {expected_version}")
    for manifest_path, package_name, cargo_version in package_versions:
        print(f"  {manifest_path}: {package_name} = {cargo_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
