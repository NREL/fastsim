# Adapted from https://github.com/NatLabRockies/routee-powertrain/blob/860db655fe98b897e25c4bd26188c0a3cb289369/docs/examples/_convert_examples_to_notebooks.py

from pathlib import Path
import nbformat
import argparse
import shutil
import sys


def script_to_notebook(script_path: Path, notebook_path: Path) -> None:
    # Read the script
    with open(script_path, "r") as script_file:
        lines = script_file.readlines()

    notebook = nbformat.v4.new_notebook()
    current_code_block: list[str] = []
    current_markdown_block: list[str] = []

    def add_code_cell(block: list[str]) -> None:
        if block and "".join(block).strip():
            notebook.cells.append(nbformat.v4.new_code_cell("".join(block).strip()))

    def add_markdown_cell(block: list[str]) -> None:
        if block:
            notebook.cells.append(nbformat.v4.new_markdown_cell("".join(block).strip()))

    in_markdown = False
    for line in lines:
        stripped = line.strip()

        # Use # %% as a code cell boundary
        if stripped.startswith("# %%"):
            add_code_cell(current_code_block)
            current_code_block = []
            continue

        # Only treat """ as markdown delimiter at top level (not indented)
        if stripped.startswith('"""') and not line[0].isspace():
            # Single-line """text""" — emit as a one-shot markdown cell
            if stripped.endswith('"""') and stripped != '"""':
                add_code_cell(current_code_block)
                current_code_block = []
                content = stripped[3:-3]
                if content:
                    add_markdown_cell([content])
            else:
                in_markdown = not in_markdown
                if in_markdown:
                    add_code_cell(current_code_block)
                    current_code_block = []
                else:
                    add_markdown_cell(current_markdown_block)
                    current_markdown_block = []
        elif in_markdown:
            current_markdown_block.append(line)
        else:
            current_code_block.append(line)

    add_code_cell(current_code_block)
    add_markdown_cell(current_markdown_block)

    with open(notebook_path, "w") as notebook_file:
        nbformat.write(notebook, notebook_file)


def notebook_to_script(notebook_path: Path, script_path: Path) -> None:
    """Convert a Jupyter notebook back to a Python script with markdown in triple quotes."""
    # Read the notebook
    with open(notebook_path, "r") as notebook_file:
        notebook = nbformat.read(notebook_file, as_version=4)

    script_lines = []

    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            # Add markdown content wrapped in triple quotes
            script_lines.append('"""\n')
            script_lines.append(cell.source)
            if not cell.source.endswith("\n"):
                script_lines.append("\n")
            script_lines.append('"""\n')
        elif cell.cell_type == "code":
            # Add code content directly
            if cell.source.strip():  # Only add non-empty code cells
                script_lines.append(cell.source)
                if not cell.source.endswith("\n"):
                    script_lines.append("\n")

    # Write the script
    with open(script_path, "w") as script_file:
        script_file.writelines(script_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert between Python scripts and Jupyter notebooks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all example scripts to notebooks (default behavior)
  python demos_to_notebooks.py

  # Convert all example notebooks back to scripts  
  python demos_to_notebooks.py --to-script

  # Convert specific file to notebook
  python demos_to_notebooks.py --file my_example.py

  # Convert specific notebook to script
  python demos_to_notebooks.py --file my_example.ipynb --to-script
        """,
    )

    parser.add_argument(
        "--to-script",
        action="store_true",
        help="Convert notebooks to scripts (default: convert scripts to notebooks)",
    )
    parser.add_argument(
        "--file", type=Path, help="Convert a specific file instead of all examples"
    )
    parser.add_argument(
        "--dir", type=Path, help="Find examples in this directory", default=Path(__file__).parent
    )
    parser.add_argument(
        "--out-dir", type=Path, help="Output directory for converted files", default=Path(__file__).parent/"_demo_notebooks"
    )

    args = parser.parse_args()
    
    # Clear output directory
    shutil.rmtree(args.out_dir, ignore_errors=True)
    args.out_dir.mkdir()

    if args.file:
        # Convert specific file
        input_file = args.file
        if not input_file.is_absolute():
            input_file = args.dir / input_file

        if not input_file.exists():
            print(f"Error: File {input_file} does not exist")
            sys.exit(1)

        rel_path = input_file.relative_to(args.dir)
        out_path = args.out_dir / rel_path

        if args.to_script:
            if input_file.suffix != ".ipynb":
                print(f"Error: {input_file} is not a notebook file")
                sys.exit(1)
            out_path = out_path.with_suffix(".py")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            notebook_to_script(input_file, out_path)
            print(f"Converted {input_file} to {out_path}")
        else:
            if input_file.suffix != ".py":
                print(f"Error: {input_file} is not a Python file")
                sys.exit(1)
            out_path = out_path.with_suffix(".ipynb")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            script_to_notebook(input_file, out_path)
            print(f"Converted {input_file} to {out_path}")
    else:
        # Convert all example files (recursively through subdirectories)
        if args.to_script:
            notebooks = sorted(args.dir.rglob("demo*.ipynb"))
            converted_count = 0
            for notebook in notebooks:
                rel_path = notebook.relative_to(args.dir)
                out_path = (args.out_dir / rel_path).with_suffix(".py")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                notebook_to_script(notebook, out_path)
                print(f"Converted {rel_path} to {out_path}")
                converted_count += 1

            if converted_count == 0:
                print("No example notebooks found to convert")
        else:
            scripts = sorted(args.dir.rglob("demo*.py"))
            converted_count = 0
            for script in scripts:
                rel_path = script.relative_to(args.dir)
                out_path = (args.out_dir / rel_path).with_suffix(".ipynb")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                script_to_notebook(script, out_path)
                print(f"Converted {rel_path} to {out_path}")
                converted_count += 1

            if converted_count == 0:
                print("No example scripts found to convert")
