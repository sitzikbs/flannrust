//! CLI wrapper around `xval::render` -- turns the JSON emitted by
//! `examples/report_data.rs` into a self-contained HTML scorecard on
//! stdout. All the actual parsing/rendering logic lives in
//! `xval::report::render` (unit-tested in `tests/render_report_test.rs`
//! against a handcrafted JSON fixture); this binary is just argument
//! handling and I/O.
//!
//! Usage:
//! ```text
//! cargo run -p xval --release --example report_data > report.json
//! cargo run -p xval --release --example render_report -- report.json > report.html
//! ```
//! Pass `-` as the path argument to read the JSON from stdin instead of a
//! file, e.g. to pipe `report_data`'s stdout straight through without an
//! intermediate file:
//! ```text
//! cargo run -p xval --release --example report_data | cargo run -p xval --release --example render_report -- - > report.html
//! ```

use std::io::Read;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: render_report <report.json|->");
        std::process::exit(2);
    });

    let json = if path == "-" {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf).expect("failed to read JSON from stdin");
        buf
    } else {
        std::fs::read_to_string(&path).unwrap_or_else(|e| {
            eprintln!("render_report: failed to read {path}: {e}");
            std::process::exit(1);
        })
    };

    match xval::render(&json) {
        Ok(html) => {
            print!("{html}");
        }
        Err(e) => {
            eprintln!("render_report: {e}");
            std::process::exit(1);
        }
    }
}
