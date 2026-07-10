# Updating the Docs

This documentation is built using [Jupyter Book v2](https://mystmd.org/) (MyST).



## Setup

Install the doc build dependencies:

- Using Pixi:
  ```bash
  pixi install -e docs
  ```

- Manually:
  ```bash
  pip install --group docs
  ```

## Local Development

Convert the demo scripts to notebooks and launch the jupyter book, executing notebooks

- Using Pixi:
  ```bash
  pixi run docs
  ```

- Manually:
  ```bash
  cd docs

  # Convert demo scripts to notebooks
  python demos_to_notebooks.py

  # Start the dev server with live reload
  # The `--execute` flag runs the demo notebooks and populates their
  # outputs, so plots appear in the rendered pages:
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
  the existing convention
  - Markdown cells are surrounded by triple quotes
    ```python
    """
    This becomes a **markdown cell**
    """
    ```

  - Code cells start with `# %%`
    ```python
    # %%
    # This becomes a code cell
    
    # %%
    ```
    - Code cells can also use the notebook tags (described [here](https://jupyterbook.org/v1/interactive/hiding.html)) with these lines:
      - `# notebook: hide-input`
      - `# notebook: remove-input`
      - `# notebook: hide-output`
      - `# notebook: remove-output`
      - `# notebook: hide-cell`
      - `# notebook: remove-cell`

  - Then regenerate notebooks:
    ```bash
    python demos_to_notebooks.py
    ```
    - Only files matching `demo*.py` are converted. Other files like `plot_utils.py` and `test_demos.py` are ignored by the converter.

- **Table of Contents**: Edit `docs/myst.yml` under `project.toc`

## Publishing

Pushing to the `fastsim-3` branch triggers a GitHub Actions workflow
(`deploy_docs.yaml`) that builds and deploys the book to GitHub Pages.
