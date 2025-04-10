import fastsim
import ast
import sys
import os
import argparse

# NOTE: REQUIRES PYTHON 3.9 OR NEWER!
# https://docs.python.org/3/library/ast.html#ast.unparse


class DocCopier(ast.NodeTransformer):
    @staticmethod
    def make_docstring(node: ast.AST, docstring: str) -> ast.AST:
        # Only make docstring if not already present
        if ast.get_docstring(node) is None:
            docstring_node = ast.Expr(value=ast.Str(docstring))
            node.body.insert(0, docstring_node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        # Make docstring for class
        class_object = getattr(fastsim, node.name)
        class_docstring = class_object.__doc__
        if class_docstring is not None:
            node = self.make_docstring(node, class_docstring)
        # Make docstrings for all child methods
        for child_node in ast.iter_child_nodes(node):
            if isinstance(child_node, ast.FunctionDef) and not child_node.name.startswith("_"):
                child_docstring = getattr(class_object, child_node.name).__doc__
                if child_docstring is not None:
                    child_node = self.make_docstring(child_node, child_docstring)
        self.generic_visit(node)
        return node


ARG_DEFAULTS = {
    "pyi_filepath": "fastsim-py/python/fastsimfastsim.pyi",
    "overwrite": False,
    "output_suffix": "_output",
    "backup_suffix": "_backup",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Programmatically copy Rust docstrings into .pyi file"
    )
    parser.add_argument(
        "pyi_filepath",
        nargs="?",
        type=str,
        default=ARG_DEFAULTS["pyi_filepath"],
        help=".pyi file to copy docstrings into (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_false" if ARG_DEFAULTS["overwrite"] else "store_true",
        help="overwrite original .pyi file, saving backup with backup-suffix in filename (default: %(default)s)",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=ARG_DEFAULTS["output_suffix"],
        help="suffix to use for output .pyi file (default: %(default)s)",
    )
    parser.add_argument(
        "--backup-suffix",
        type=str,
        default=ARG_DEFAULTS["backup_suffix"],
        help="suffix to use for renamed .pyi file (default: %(default)s)",
    )
    args = parser.parse_args()
    if not args.pyi_filepath.endswith(".pyi"):
        parser.error("file extension must be .pyi")
    return args


if __name__ == "__main__":
    # Check Python version
    assert sys.version_info >= (3, 9, 0), "Python version must be 3.9 or newer to use ast.unparse()"

    # Parse arguments
    args = parse_args()

    filename, extension = os.path.splitext(args.pyi_filepath)
    if args.overwrite:
        pyi_backup_filepath = filename + args.backup_suffix + extension
    else:
        pyi_output_filepath = filename + args.output_suffix + extension

    # Read .pyi file as AST tree
    with open(args.pyi_filepath) as f:
        code = f.read()
    tree = ast.parse(code)

    # Magical bits
    ast.fix_missing_locations(DocCopier().visit(tree))

    # Write output .pyi file
    if args.overwrite:
        # Rename original .pyi file
        os.rename(args.pyi_filepath, pyi_backup_filepath)
        output_filepath = args.pyi_filepath
    else:
        output_filepath = pyi_output_filepath
    with open(output_filepath, "w") as f:
        f.write(ast.unparse(tree))
