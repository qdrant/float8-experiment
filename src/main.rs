use serde::{Deserialize, Serialize};
use clap::Parser;
use std::cmp::Ordering;
use std::fs::File;
use std::io::BufReader;
use std::io::BufRead;
use std::collections::HashSet;
use rayon::prelude::*;

mod pq;



#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the input file containing experiment data
    #[arg(short, long)]
    input: String,
    
    /// Path to the NPY file containing vectors
    #[arg(short, long)]
    vectors: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Example {
    query: Vec<f32>,
    closest_ids: HashSet<usize>,
    closest_scores: Vec<f32>,
}




struct Experiment {
    examples: Vec<Example>
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

fn noramlize_vectors(vectors: &mut Vec<Vec<f32>>) {
    for vector in vectors {
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in vector {
            *x = *x / norm;
        }
    }
}

fn into_float8(vectors: Vec<Vec<f32>>) -> Vec<Vec<float8::F8E4M3>> {
    let mut float8_vectors = Vec::new();
    for vector in vectors {
        let mut float8_vector = Vec::new();
        for x in vector {
            float8_vector.push(float8::F8E4M3::from_f32(x));
        }
        float8_vectors.push(float8_vector);
    }
    float8_vectors
}

fn dot_product(qeury: &[f32], stored: &[float8::F8E4M3]) -> f32 {
    let mut result = 0.0;
    for (q, s) in qeury.iter().zip(stored.iter()) {
        result += q * s.to_f32();
    }
    result
}

struct ScoredVector {
    id: usize,
    score: f32,
}

impl PartialEq for ScoredVector {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for ScoredVector {}

impl PartialOrd for ScoredVector {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl Ord for ScoredVector {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.partial_cmp(&other.score).unwrap()
    }
}

fn knn_search(query: &[f32], vectors: &[Vec<float8::F8E4M3>], k: usize) -> Vec<ScoredVector> {
    let mut pq = pq::FixedLengthPriorityQueue::new(k);
    for (i, vector) in vectors.iter().enumerate() {
        pq.push(ScoredVector { id: i, score: dot_product(query, vector) });
    }
    pq.into_sorted_vec()
}

fn main() {
    let args = Args::parse();
    let experiment = Experiment::load_from_file(&args.input);
    println!("Loaded {} examples from {}", experiment.examples.len(), args.input);

    let bytes = std::fs::read(&args.vectors).unwrap();

    let npy = npyz::NpyFile::new(&bytes[..]).unwrap();

    let shape = npy.shape().to_vec();

    eprintln!("Shape: {:#?}", shape);

    let all_data = npy.into_vec::<f32>().unwrap();

    let mut vectors = all_data.chunks(shape[1] as _).map(|chunk| chunk.to_vec()).collect::<Vec<Vec<f32>>>();
    
    eprintln!("Vectors = {:#?}", vectors.len());

    noramlize_vectors(&mut vectors);

    let float8_vectors = into_float8(vectors);

    let (correct, total) = experiment.examples.par_iter()
        .map(|example| {
            let query = &example.query;
            let closest_ids = &example.closest_ids;
            let results = knn_search(query, &float8_vectors, closest_ids.len());
            
            let mut local_correct = 0;
            for result in results.iter() {
                if closest_ids.contains(&result.id) {
                    local_correct += 1;
                }
            }
            
            (local_correct, closest_ids.len())
        })
        .reduce(|| (0, 0), |(acc_correct, acc_total), (correct, total)| {
            (acc_correct + correct, acc_total + total)
        });

    println!("Correct = {}/{} = {}", correct, total, correct as f32 / total as f32);

}
