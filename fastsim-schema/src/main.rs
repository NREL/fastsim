use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(version, about = "fastsim-schema CLI tools", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build an index file (vehicles.jsonl) from a database directory
    BuildIndex(BuildIndexArgs),
}

#[derive(clap::Args)]
struct BuildIndexArgs {
    /// Database root directory containing versioned schema folders (v1, v2, ...)
    root: std::path::PathBuf,

    /// Database schema version
    #[arg(short, long, value_name = "SCHEMA", default_value_t = 1)]
    schema: u32,

    /// Output index file path, relative to directory ROOT/vSCHEMA.
    #[arg(
        short,
        long = "output",
        value_name = "OUTPUT",
        default_value = "vehicles.jsonl"
    )]
    output_file: std::path::PathBuf,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::BuildIndex(args) => run_build_index(args),
    }
}

fn run_build_index(args: BuildIndexArgs) -> anyhow::Result<()> {
    let schema_version = args.schema;

    let schema_dir = std::path::absolute(&args.root)?.join(format!("v{}", schema_version));
    anyhow::ensure!(
        schema_dir.exists(),
        "directory not found at schema v{} path: {:?}",
        schema_version,
        schema_dir
    );
    anyhow::ensure!(
        schema_dir.is_dir(),
        "resolved schema v{} path is not a directory: {:?}",
        schema_version,
        schema_dir
    );
    let schema_dir = std::fs::canonicalize(schema_dir)?;

    let output_file_path = {
        let candidate = std::path::PathBuf::from(&args.output_file);
        if candidate.is_absolute() {
            candidate
        } else {
            schema_dir.join(candidate)
        }
    };
    anyhow::ensure!(
        output_file_path.extension() == Some(std::ffi::OsStr::new("jsonl")),
        "output file path must have a .jsonl extension: {:?}",
        output_file_path
    );

    println!("Building index from directory {}", schema_dir.display());

    let schema_prefix = std::path::PathBuf::from(format!("v{}", schema_version));
    let mut entries = Vec::new();
    for entry in walkdir::WalkDir::new(&schema_dir) {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }
        let relative_path = entry.path().strip_prefix(&schema_dir).unwrap();
        let schema_file_path = schema_prefix.join(relative_path);
        let Some(schema_file_path_str) = schema_file_path.to_str() else {
            println!("Skipping non-UTF8 path: {:?}", schema_file_path);
            continue;
        };
        if !schema_file_path_str.ends_with(".yaml") {
            println!("Skipping non-YAML file: {:?}", schema_file_path_str);
            continue;
        }
        match schema_file_path_str.parse::<fastsim_schema::IndexEntryV1>() {
            Ok(entry) => entries.push(entry),
            Err(_) => println!("Skipping unparsable file: {:?}", schema_file_path_str),
        }
    }
    println!("Created {} index entries", entries.len());

    entries.sort_by(|a, b| a.path.cmp(&b.path));

    let mut output_file = std::fs::File::create(&output_file_path)?;
    match schema_version {
        1 => fastsim_schema::write_jsonl_v1(&mut output_file, &entries)?,
        _ => anyhow::bail!("unsupported schema version: {}", schema_version),
    }
    println!("Index saved to {}", output_file_path.display());

    Ok(())
}
