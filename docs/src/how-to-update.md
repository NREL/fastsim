# How to Update This Markdown Book

[mdBook Documentation](https://rust-lang.github.io/mdBook/)

## Setup

1. If not already done, [install mdbook](https://rust-lang.github.io/mdBook/guide/installation.html)

## Publishing

1. Update `book.toml` or files in `docs/src/`
1. Make sure the docs look good locally: `mdbook build docs/ --open`
1. Commit files and push to `main` branch

After that, a GitHub action will build the book and publish it [here](https://pages.github.nrel.gov/MBAP/mbap-computing/)
