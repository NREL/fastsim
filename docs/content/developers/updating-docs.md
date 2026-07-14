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
  content/            Markdown pages and notebooks
  assets/             Images and icons
  myst.yml            Jupyter Book configuration and table of contents
```

## Adding or Modifying Pages

1. Add `.md` and/or `.ipynb` files to `docs/content/`

1. Edit the table of contents `docs/myst.yml` to reflect the updated directory structure

## Publishing

Pushing to the `fastsim-3` branch triggers a GitHub Actions workflow
(`deploy_docs.yaml`) that builds and deploys the book to GitHub Pages.
