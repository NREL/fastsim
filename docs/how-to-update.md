# How to Update This Book

This documentation is built using [Jupyter Book v2](https://mystmd.org/) (MyST).

## Setup

Install the doc build dependencies:

```bash
pip install -r docs/requirements.txt
```

## Local Development

From the `docs/` directory:

```bash
# Convert demo scripts to notebooks
python demos_to_notebooks.py --dir ../python/fastsim/demos

# Start the dev server with live reload
jupyter book start
```

The site will be available at `http://localhost:3000`.

## Building Static HTML

```bash
cd docs
jupyter book build --html
```

Output is written to `docs/_build/html/`.

## Adding or Modifying Pages

- **Markdown pages**: Add `.md` files to `docs/` and reference them in the `toc:` section of `myst.yml`
- **Demo notebooks**: Add demo scripts to `python/fastsim/demos/` following the `"""` markdown block convention, then run `demos_to_notebooks.py` to generate notebooks
- **Table of Contents**: Edit `docs/myst.yml` under `project.toc:`

## Publishing

Pushing to the `fastsim-3` branch triggers a GitHub Actions workflow (`deploy_docs.yaml`) that builds and deploys the book to GitHub Pages.
