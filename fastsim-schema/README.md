# fastsim-schema

Schema and tooling crate for FASTSim vehicle database structures.

This crate is used by downstream data tooling and can be released independently of
the main FASTSim simulation crates.

## Releasing

1. Bump the crate version in `fastsim-schema/Cargo.toml`.
2. Update the workspace dependency version in `Cargo.toml` for `fastsim-schema`.
3. Run version checks from the repository root:

	 ```sh
	 python python/fastsim/check_versions.py
	 ```

	 Or, via Pixi:

	 ```sh
	 pixi run check-versions
	 ```

4. Publish:

	 ```sh
	 cargo publish --manifest-path fastsim-schema/Cargo.toml --locked
	 ```

## Why This Is Not Lockstep-Versioned With FASTSim

`fastsim-schema` has a different compatibility surface and release cadence than `fastsim-core`.

- Schema updates may happen for data/modeling workflows that do not require a
	simulation-engine release.
- FASTSim core releases may include simulation/runtime changes that do not
	require a schema bump.

Keeping versions independent allows smaller, targeted releases while still
pinning explicit dependency versions in the workspace for reproducibility.

