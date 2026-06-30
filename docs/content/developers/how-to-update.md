# How to Update This Book

This documentation is built using [Jupyter Book v2](https://mystmd.org/) (MyST).

## Setup

Install the doc build dependencies:

```bash
cd fastsim/docs
pip install -r requirements.txt
```

## Local Development

From the `docs/` directory, convert the demo scripts to notebooks and
start the dev server:

```bash
# Convert demo scripts to notebooks
python demos_to_notebooks.py

# Start the dev server with live reload
jupyter book start
```

The `--execute` flag runs the demo notebooks and populates their
outputs, so plots appear in the rendered pages:

```bash
jupyter book start --execute
```

The site will be available at `http://localhost:3000`.

## Building Static HTML

```bash
jupyter book build --html --execute
```

Output is written to `docs/_build/html/`.

## Directory Layout

```
docs/
  content/            Markdown pages (intro, calibration, developer guides)
  demo_scripts/       Source Python scripts for demos
  demo_notebooks/     Generated Jupyter notebooks (do not edit directly)
  assets/             Images and icons
  myst.yml            Jupyter Book configuration and table of contents
  demos_to_notebooks.py   Script to convert demo_scripts/ to demo_notebooks/
```

## Adding or Modifying Pages

- **Markdown pages**: Add `.md` files to `docs/content/` and reference
  them in the `toc:` section of `myst.yml`
- **Demo notebooks**: Add demo scripts to `docs/demo_scripts/` following
  the existing convention (`"""` blocks become markdown cells, `# %%`
  marks code cell boundaries). Then regenerate notebooks:
  ```bash
  python demos_to_notebooks.py
  ```
  Only files matching `demo*.py` are converted. Other files like
  `plot_utils.py` and `test_demos.py` are ignored by the converter.
- **Table of Contents**: Edit `docs/myst.yml` under `project.toc:`

## Publishing

Pushing to the `fastsim-3` branch triggers a GitHub Actions workflow
(`deploy_docs.yaml`) that builds and deploys the book to GitHub Pages.
