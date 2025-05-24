use serde::{Deserialize, Serialize};
use clap::Parser;
use std::fs::File;
use std::io::BufReader;
use std::io::BufRead;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the input file containing experiment data
    #[arg(short, long)]
    input: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Example {
    query: Vec<f32>,
    closest_ids: Vec<usize>,
    closest_scores: Vec<f32>,
}

struct Experiment {
    examples: Vec<Example>,
}

impl Experiment {
    pub fn load_from_file(path: &str) -> Self {
        // Load from jsonl file
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);
        // Read line by line
        let mut examples = Vec::new();
        for line in reader.lines() {
            let example: Example = serde_json::from_str(&line.unwrap()).unwrap();
            examples.push(example);
        }

        Self { examples }
    }
}

fn main() {
    let args = Args::parse();
    let experiment = Experiment::load_from_file(&args.input);
    println!("Loaded {} examples from {}", experiment.examples.len(), args.input);
}
