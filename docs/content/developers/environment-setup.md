# Environment Setup

A few options exist for environment setup:
1. [Pixi (recommended)](#1-pixi)
2. [Anaconda](#2-anaconda)
2. [Custom](#3-custom)

## 1. Pixi
[Pixi](https://pixi.prefix.dev/latest/) is a tool that ensures environment hygiene and reproducibility. FASTSim has an established setup for 
Pixi that you can take advantage of to quickly get started.

### Pixi Installation

Full installation instructions are available in the pixi docs:
[https://pixi.prefix.dev/latest/installation/](https://pixi.prefix.dev/latest/installation/)

After installation, confirm pixi is available in a fresh terminal:

```sh
pixi --version
```

### Environment Installation

From the repository root, install FASTSim and development dependencies:

```sh
pixi install -e dev
```

The `dev` Pixi environment installs FASTSim, the Rust compiler, testing dependencies, and other conveniences. If these are not necessary, see the `pyproject.toml` file for other options.

### Environment Usage

To use the environment in a terminal:

- Run individual commands:

```sh
pixi run -e dev <command>
```

- Or, activate an interactive shell:

```sh
pixi shell -e dev
```

See the Pixi documentation or run `pixi --help` for more usage information. The [direnv](https://direnv.net) tool automates launching a shell for you, and FASTSim has a preconfigured `.envrc` file. 

### direnv: Automatic Environment Activation

If you use direnv, you can auto-load the pixi dev environment when entering this repository.

Setup:

1. Install direnv: [https://direnv.net/docs/installation.html](https://direnv.net/docs/installation.html)
2. Enable the hook for your shell (see [direnv docs](https://direnv.net/docs/hook.html)).
3. Allow `direnv` to run `.envrc`:
    ```sh
    direnv allow
    ```

After that, entering the directory will auto-activate the pixi `dev` environment. Messages about environment variables can be suppressed by creating `~/.config/direnv/direnv.toml` containing the line:
```toml
hide_env_diff = true
```

## 2. Anaconda

Anaconda users can easily install Rust using the conda-forge `rust` package

1. Create a new environment with Python and Rust (e.g. named `fastsim`)
    ```
    conda create --name fastsim python=3.12 rust -c conda-forge
    ```

2. Activate the new environment
    ```
    conda activate fastsim
    ```

2. From the repository root, install FASTSim and development dependencies:

    ```sh
    pip install --upgrade pip
    pip install --group dev -e .
    ```

:::{note}
This section suggests dependencies independent of the `pyproject.toml`, so may become outdated.

As of writing, a way to build an Anaconda environment from `pixi.lock` using
the [`conda-lockfiles` plugin](https://github.com/conda/conda-lockfiles) is nearly supported:

```console
conda install --name base conda-forge::conda-lockfiles
```

```
conda env create --name fastsim --file pixi.lock
```

However, this plugin only supports `pixi.lock` version 6, while version 7 released in [May 2026](https://pixi.prefix.dev/latest/CHANGELOG/#0680-2026-05-07).

Output:
```
PluginError: Failed to parse environment specification from file: field 'version': Input should be less than or equal to 6
```

Relevant links:
- https://github.com/conda/conda-lockfiles/pull/143
- https://github.com/conda/conda-lockfiles/issues/44
:::


## 3. Custom

If you do not want to use Pixi or Anaconda, install Rust and Python dependencies manually.

1. Install Rust system-wide: [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install)
2. Create and activate a Python environment (using your environment manager of choice).
3. From the repository root, install FASTSim and development dependencies:

    ```sh
    pip install --group dev -e .
    ```
