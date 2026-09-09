//! Build a static kd-tree, then run a knn and a radius query.
//!
//! Run with: `cargo run --example knn`

use flannrust::{ConstDim, KdTreeBuilder, ResultItem};

fn main() {
    // 1000 deterministic pseudo-random 3-D points in [-10, 10)^3
    // (xorshift64 -- no dependencies needed for an example).
    let mut state: u64 = 42;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 * 20.0 - 10.0
    };
    let pts: Vec<[f64; 3]> = (0..1000).map(|_| [next(), next(), next()]).collect();

    let tree = KdTreeBuilder::new(ConstDim::<3>, pts.as_slice()).build();

    // 5 nearest neighbors of the origin. NOTE: distances are SQUARED for L2.
    let query = [0.0, 0.0, 0.0];
    let mut indices = [0u32; 5];
    let mut dists = [0.0f64; 5];
    let found = tree.knn_search(&query, &mut indices, &mut dists);
    println!("knn: {found} nearest neighbors of {query:?}:");
    for (i, d) in indices.iter().zip(&dists) {
        println!("  index {i:4}  squared distance {d:.6}");
    }

    // Every point with SQUARED distance < 4.0 (i.e. within radius 2.0),
    // sorted by distance. Strictly `dist < radius`.
    let mut out: Vec<ResultItem<u32, f64>> = Vec::new();
    let n = tree.radius_search(&query, 4.0, &mut out);
    println!("radius: {n} points within squared distance 4.0");
    for item in out.iter().take(5) {
        println!(
            "  index {:4}  squared distance {:.6}",
            item.index, item.distance
        );
    }
}
