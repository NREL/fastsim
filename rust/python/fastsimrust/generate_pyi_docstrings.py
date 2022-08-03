import fastsim
import ast
import shutil
import sys
import os

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
            if isinstance(child_node, ast.FunctionDef):
                child_docstring = getattr(class_object, child_node.name).__doc__
                if child_docstring is not None:
                    child_node = self.make_docstring(child_node, child_docstring)
        self.generic_visit(node)
        return node


if __name__ == "__main__":
    # Check Python version
    assert sys.version_info >= (3,9,0), "Python version must be 3.9 or newer to use ast.unparse()"

    # If True, backup old .pyi file and save output over original
    # If False, save output .pyi file with OUTPUT_SUFFIX
    OVERWRITE_AND_BACKUP = False

    PYI_FILEPATH = "./rust/python/fastsimrust/fastsimrust.pyi"
    FILENAME, EXTENSION = os.path.splitext(PYI_FILEPATH)
    if OVERWRITE_AND_BACKUP: 
        BACKUP_SUFFIX = "_backup"
        PYI_BACKUP_FILEPATH = FILENAME + BACKUP_SUFFIX + EXTENSION
    else:
        OUTPUT_SUFFIX = "_output"
        PYI_OUTPUT_FILEPATH = FILENAME + OUTPUT_SUFFIX + EXTENSION

    # Read .pyi file as AST tree
    with open(PYI_FILEPATH) as f:
        code = f.read()
    tree = ast.parse(code)

    # Magical bits
    ast.fix_missing_locations(DocCopier().visit(tree))

    # Write output .pyi file
    if OVERWRITE_AND_BACKUP:
        # Copy input .pyi file as backup
        shutil.copy(PYI_FILEPATH, PYI_BACKUP_FILEPATH)
        output_filepath = PYI_FILEPATH
    else:
        output_filepath = PYI_OUTPUT_FILEPATH
    with open(output_filepath, "w") as f:
        f.write(ast.unparse(tree))
