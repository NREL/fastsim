# Updating the Docs

FASTSim's documentation is built using [Jupyter Book](https://jupyterbook.org/) v2, which is built upon the [MyST Document Engine](https://mystmd.org/).

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

Launch Jupyter Book with live reload and notebook execution.

- Using Pixi:
  ```bash
  pixi run docs
  ```

- Manually:
  1. Change to the `docs/` directory

      ```bash
      cd docs
      ```
  1. Start the dev server with live reload:

      ```bash
      jupyter book start --execute
      ```

      The `--execute` flag runs the demo notebooks and populates their outputs,
      allowing plots to appear in the rendered pages

The site will be available at `http://localhost:3000`

## Building Static HTML

```bash
jupyter book build --strict --html --execute
```

Output is written to `docs/_build/html/`.
The `--strict` flag checks for broken internal references.

## Directory Layout

```
docs/
  content/            Markdown pages and notebooks
    old/              Outdated demo notebooks (still executed via pytest for additional coverage)
    ...
  assets/             Images and icons
  myst.yml            Jupyter Book configuration and table of contents
```

## Adding or Modifying Pages

1. Add `.md` and/or `.ipynb` files to `docs/content/`

1. Edit the table of contents `docs/myst.yml` to reflect the updated doc structure

1. Check that a strict docs build runs cleanly:

    `jupyter book build --strict --html --execute`

## Publishing

Pushing to the `fastsim-3` branch triggers a GitHub Actions workflow
(`deploy_docs.yaml`) that builds and deploys the book to GitHub Pages.
