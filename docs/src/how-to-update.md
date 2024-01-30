# How to Update This Markdown Book

[mdBook Documentation](https://rust-lang.github.io/mdBook/)

## Setup

If not already done, [install mdbook](https://rust-lang.github.io/mdBook/guide/installation.html)

## Serving locally
Run the following in the repository root directory: 
1. If any python files were modified, 
    1. Install [pipx](https://github.com/pypa/pipx#install-pipx)
    1. Install [pydoc-markdown](https://niklasrosenstein.github.io/pydoc-markdown/#installation-)
    1. run `pydoc-markdown -I python/ --render-toc > docs/src/python-doc.md`. Do not modify this file manually. 
1. Run `mdbook serve --open docs/`

## Publishing
1. Update `book.toml` or files in `docs/src/`
1. Make sure the docs look good locally: `mdbook build docs/ --open`
1. Commit files and push to `fastsim-2` branch

After that, a GitHub action will build the book and publish it [here](https://pages.github.nrel.gov/MBAP/mbap-computing/)