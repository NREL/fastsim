import fastsim as fsim
from pathlib import Path

from pydoc_markdown.interfaces import Context
from pydoc_markdown.contrib.loaders.python import PythonLoader
from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer

def main():
    context = Context(directory=fsim.package_root())
    loader = PythonLoader(
        search_path=[fsim.package_root()],
        ignore_when_discovered=["test", "tests", "setup", "demos"]
    )
    renderer = MarkdownRenderer(
        # TODO: figure out what options in here will produce same result as
        # `pydoc-markdown -I python --render-toc > docs/src/python-doc.md`  
        # # Commented args
        # render_toc=True,  
        # render_toc_title=False,  
        # toc_maxdepth=1,  
    )

    loader.init(context)
    renderer.init(context)

    modules = loader.load()

    with open(fsim.package_root() / "../../docs/src/python-doc.md", "w") as f:
        f.write(renderer.render_to_string(modules))

