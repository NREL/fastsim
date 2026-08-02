#!/usr/bin/env python3

"""Check version consistency for selected FASTSim Rust packages.

Reads the FASTSim project version from `pyproject.toml` and compares it to
the versions declared in Cargo manifests for:
- fastsim-py
- fastsim-core
- fastsim-proc-macros

Also verifies that the `fastsim-proc-macros` dependency in
`fastsim-core/Cargo.toml` and workspace manifests are configured correctly.
"""

import json
import os
import pathlib
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:
    print("error: python3 with tomllib support (3.11+) is required", file=sys.stderr)
    sys.exit(2)


TARGET_PACKAGES = {"fastsim-py", "fastsim-core", "fastsim-proc-macros"}


def supports_color(stream: object) -> bool:
    """Return True when ANSI colors should be emitted for this stream."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


def colorize(text: str, color: str, enabled: bool) -> str:
    """Wrap text in ANSI color escapes when color output is enabled."""
    if not enabled:
        return text

    colors = {
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "reset": "\033[0m",
    }
    return f"{colors[color]}{text}{colors['reset']}"


def load_pyproject_version(pyproject_path: pathlib.Path) -> str:
    """Return the project version declared in pyproject.toml."""
    pyproject = load_toml(pyproject_path)
    return pyproject["project"]["version"]


def load_toml(path: pathlib.Path) -> dict:
    """Load a TOML file from disk and return it as a dictionary."""
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_workspace_packages(
    repo_root: pathlib.Path,
) -> tuple[pathlib.Path, list[tuple[str, str, str]]]:
    """Return the workspace root and a sorted list of (manifest_path, name, version) tuples."""
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
        if package["name"] not in TARGET_PACKAGES:
            continue

        manifest_path = pathlib.Path(package["manifest_path"]).resolve()
        try:
            relative_manifest = manifest_path.relative_to(workspace_root)
        except ValueError:
            relative_manifest = manifest_path

        package_versions.append(
            (str(relative_manifest), package["name"], package["version"]),
        )

    return workspace_root, sorted(package_versions)


def check_fastsim_proc_macros_dependency(
    repo_root: pathlib.Path,
) -> list[str]:
    """Validate fastsim-core dependency uses workspace-managed fastsim-proc-macros."""
    core_manifest_path = repo_root / "fastsim-core" / "Cargo.toml"
    workspace_manifest_path = repo_root / "Cargo.toml"
    proc_macros_manifest_path = repo_root / "fastsim-core" / "fastsim-proc-macros" / "Cargo.toml"

    core_manifest = load_toml(core_manifest_path)
    workspace_manifest = load_toml(workspace_manifest_path)
    proc_macros_manifest = load_toml(proc_macros_manifest_path)

    core_dependency = core_manifest.get("dependencies", {}).get("fastsim-proc-macros")
    workspace_dependency = (
        workspace_manifest.get("workspace", {})
        .get("dependencies", {})
        .get(
            "fastsim-proc-macros",
        )
    )
    proc_macros_package_version = proc_macros_manifest.get("package", {}).get("version")

    problems = []

    if core_dependency != {"workspace": True}:
        problems.append(
            "fastsim-core/Cargo.toml: dependency 'fastsim-proc-macros' must set workspace = true",
        )

    if not isinstance(workspace_dependency, dict):
        problems.append(
            "Cargo.toml: workspace dependency 'fastsim-proc-macros' must be "
            "declared as an inline table",
        )
        return problems

    actual_path = workspace_dependency.get("path")
    actual_version = workspace_dependency.get("version")

    if actual_path != "fastsim-core/fastsim-proc-macros":
        problems.append(
            "Cargo.toml: workspace dependency 'fastsim-proc-macros' must set "
            'path = "fastsim-core/fastsim-proc-macros"',
        )

    if not isinstance(proc_macros_package_version, str):
        problems.append(
            "fastsim-core/fastsim-proc-macros/Cargo.toml: package.version must be set to a string",
        )
    elif actual_version != proc_macros_package_version:
        problems.append(
            "Cargo.toml: workspace dependency 'fastsim-proc-macros' must set "
            f'version = "{proc_macros_package_version}"',
        )

    return problems


def check_workspace_fastsim_core_dependency(
    repo_root: pathlib.Path,
    expected_version: str,
) -> list[str]:
    """Validate workspace dependency declaration for fastsim-core."""
    manifest_path = repo_root / "Cargo.toml"
    manifest = load_toml(manifest_path)
    workspace_dependencies = manifest.get("workspace", {}).get("dependencies", {})
    fastsim_core_dependency = workspace_dependencies.get("fastsim-core")

    if not isinstance(fastsim_core_dependency, dict):
        return [
            "Cargo.toml: workspace dependency 'fastsim-core' must be declared as an inline table",
        ]

    problems = []
    actual_path = fastsim_core_dependency.get("path")
    actual_version = fastsim_core_dependency.get("version")

    if actual_path != "fastsim-core":
        problems.append(
            "Cargo.toml: workspace dependency 'fastsim-core' must set path = \"fastsim-core\"",
        )

    if actual_version != expected_version:
        problems.append(
            "Cargo.toml: workspace dependency 'fastsim-core' must set "
            f'version = "{expected_version}"',
        )

    return problems


def check_workspace_fastsim_schema_dependency(repo_root: pathlib.Path) -> list[str]:
    """Validate workspace dependency declaration for fastsim-schema."""
    workspace_manifest_path = repo_root / "Cargo.toml"
    schema_manifest_path = repo_root / "fastsim-schema" / "Cargo.toml"

    workspace_manifest = load_toml(workspace_manifest_path)
    schema_manifest = load_toml(schema_manifest_path)

    workspace_dependencies = workspace_manifest.get("workspace", {}).get("dependencies", {})
    fastsim_schema_dependency = workspace_dependencies.get("fastsim-schema")
    schema_package_version = schema_manifest.get("package", {}).get("version")

    if not isinstance(fastsim_schema_dependency, dict):
        return [
            "Cargo.toml: workspace dependency 'fastsim-schema' must be declared as an inline table",
        ]

    problems = []
    actual_path = fastsim_schema_dependency.get("path")
    actual_version = fastsim_schema_dependency.get("version")

    if actual_path != "fastsim-schema":
        problems.append(
            "Cargo.toml: workspace dependency 'fastsim-schema' must set path = \"fastsim-schema\"",
        )

    if not isinstance(schema_package_version, str):
        problems.append(
            "fastsim-schema/Cargo.toml: package.version must be set to a string",
        )
    elif actual_version != schema_package_version:
        problems.append(
            "Cargo.toml: workspace dependency 'fastsim-schema' must set "
            f'version = "{schema_package_version}"',
        )

    return problems


def main() -> int:
    """Run version checks and print a summary or mismatch details."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    pyproject_path = repo_root / "pyproject.toml"
    expected_version = load_pyproject_version(pyproject_path)
    _workspace_root, package_versions = load_workspace_packages(repo_root)

    package_problems = []
    for manifest_path, package_name, cargo_version in package_versions:
        if cargo_version != expected_version:
            package_problems.append(
                f"{manifest_path}: package {package_name!r} has version"
                f" {cargo_version}, expected {expected_version}",
            )

    proc_macros_problems = check_fastsim_proc_macros_dependency(repo_root)
    workspace_core_problems = check_workspace_fastsim_core_dependency(repo_root, expected_version)
    workspace_schema_problems = check_workspace_fastsim_schema_dependency(repo_root)

    core_manifest = load_toml(repo_root / "fastsim-core" / "Cargo.toml")
    core_proc_macros_dependency = core_manifest.get("dependencies", {}).get(
        "fastsim-proc-macros",
        {},
    )

    workspace_manifest = load_toml(repo_root / "Cargo.toml")
    workspace_dependencies = workspace_manifest.get("workspace", {}).get("dependencies", {})
    workspace_core_dependency = workspace_dependencies.get("fastsim-core", {})
    workspace_proc_macros_dependency = workspace_dependencies.get("fastsim-proc-macros", {})
    workspace_schema_dependency = workspace_dependencies.get("fastsim-schema", {})

    schema_manifest = load_toml(repo_root / "fastsim-schema" / "Cargo.toml")
    schema_package_version = schema_manifest.get("package", {}).get("version")
    proc_macros_manifest = load_toml(
        repo_root / "fastsim-core" / "fastsim-proc-macros" / "Cargo.toml"
    )
    proc_macros_package_version = proc_macros_manifest.get("package", {}).get("version")
    proc_macros_workspace_path = workspace_proc_macros_dependency.get("path")
    proc_macros_workspace_version = workspace_proc_macros_dependency.get("version")
    core_proc_macros_source = (
        "workspace = true"
        if core_proc_macros_dependency == {"workspace": True}
        else str(core_proc_macros_dependency)
    )

    package_check_name = "package versions match pyproject.toml"
    proc_macros_check_name = "fastsim-proc-macros dependency wiring"
    workspace_core_check_name = "workspace fastsim-core path/version"
    workspace_schema_check_name = "workspace fastsim-schema path/version"
    check_sources = {
        package_check_name: [
            "pyproject.toml",
            "fastsim-core/Cargo.toml",
            "fastsim-core/fastsim-proc-macros/Cargo.toml",
            "fastsim-py/Cargo.toml",
        ],
        proc_macros_check_name: [
            "fastsim-core/Cargo.toml",
            "Cargo.toml",
            "fastsim-core/fastsim-proc-macros/Cargo.toml",
        ],
        workspace_core_check_name: [
            "Cargo.toml",
            "pyproject.toml",
        ],
        workspace_schema_check_name: [
            "Cargo.toml",
            "fastsim-schema/Cargo.toml",
        ],
    }
    check_results = [
        (package_check_name, package_problems),
        (proc_macros_check_name, proc_macros_problems),
        (workspace_core_check_name, workspace_core_problems),
        (workspace_schema_check_name, workspace_schema_problems),
    ]

    combined_problems = [problem for _, problems in check_results for problem in problems]
    has_problems = bool(combined_problems)
    output_stream = sys.stderr if has_problems else sys.stdout
    use_color = supports_color(output_stream)

    print(f"pyproject.toml project.version = {expected_version}", file=output_stream)
    print("Version checks:", file=output_stream)
    for check_name, problems in check_results:
        status = "PASS" if not problems else "FAIL"
        status_color = "green" if status == "PASS" else "red"
        colored_status = colorize(status, status_color, use_color)
        print(f"  [{colored_status}] {check_name}", file=output_stream)
        print("    - sources:", file=output_stream)
        for source in check_sources.get(check_name, []):
            print(f"      - {source}", file=output_stream)

        if check_name == package_check_name:
            for manifest_path, package_name, cargo_version in package_versions:
                print(
                    f"    - {manifest_path}: {package_name} = {cargo_version}",
                    file=output_stream,
                )

        if check_name == proc_macros_check_name:
            print(
                f"    - fastsim-core dependency source: {core_proc_macros_source}",
                file=output_stream,
            )
            print(
                "    - workspace entry: "
                f"path={proc_macros_workspace_path}, version={proc_macros_workspace_version}",
                file=output_stream,
            )
            print(
                f"    - proc-macro crate version: {proc_macros_package_version}",
                file=output_stream,
            )

        if check_name == workspace_core_check_name:
            print(
                "    - expected version: "
                f"{expected_version}; declared version: "
                f"{workspace_core_dependency.get('version')}",
                file=output_stream,
            )

        if check_name == workspace_schema_check_name:
            print(
                "    - expected version: "
                f"{schema_package_version}; declared version: "
                f"{workspace_schema_dependency.get('version')}",
                file=output_stream,
            )

        for problem in problems:
            print(f"    - {problem}", file=output_stream)

    if has_problems:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
