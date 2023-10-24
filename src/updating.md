# How to Update This Markdown Book

[mdBook Documentation](https://rust-lang.github.io/mdBook/)

## Setup
This assumes that the output of `git remote -v` is
```
external        git@github.com:NREL/fastsim.git (push)``
external        git@github.com:NREL/fastsim.git (fetch)
```

1. If not already done, [install mdbook](https://rust-lang.github.io/mdBook/guide/installation.html)
1. `git fetch external mdbook-src:mdbook-src`
1. `git checkout mdbook-src`

## Publishing
1. Update `book.toml` or files in `src/`
1. Commit files and push to `mdbook-src` branch
1. Run `sh publish.sh` to publish to `mdbook-publish` branch

After that, just run `sh publish.sh`

## Troubleshooting
Check that https://github.com/NREL/fastsim/settings/pages is set to deploy from the `mdbook-publish` branch.  