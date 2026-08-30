//! CLI wrapper around `xval::render` -- turns the JSON emitted by
//! `examples/report_data.rs` into a self-contained HTML scorecard on
//! stdout. All the Rust-vs-C++ parsing/rendering logic lives in
//! `xval::report::render` (unit-tested in `tests/render_report_test.rs`
//! against a handcrafted JSON fixture); this binary is just argument
//! handling and I/O -- PLUS (M-py Task 5, moved into the library at M2.6
//! task 3 for testability) an optional second "Python bindings" section,
//! rendered by `xval::render_python` from `crates/flannrust-py/python/
//! bench/bench_py.py`'s emitted JSON. That JSON is spliced into the base
//! HTML right before the `<section class="methodology">` anchor (see
//! `inject_python_section`, still local to this file -- pure HTML-string
//! splicing, no JSON involved), reusing the base template's existing CSS
//! classes (`section`/`h2`/`legend`/`pill`/`table`/`scroll-x`/`num`) -- no
//! new stylesheet rules.
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
//! An optional second argument is the Python bench JSON (see
//! `crates/flannrust-py/python/bench/bench_py.py`'s docstring for how to
//! (re)generate it); when present, a "Python bindings" section is appended:
//! ```text
//! cargo run -p xval --release --example render_report -- report.json report_py.json > report.html
//! ```
//! Omitting the second argument reproduces the exact pre-Task-5 output
//! (only `xval::render`'s own HTML, untouched).

use std::io::Read;

fn read_json_arg(path: &str) -> String {
    if path == "-" {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf).expect("failed to read JSON from stdin");
        buf
    } else {
        std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("render_report: failed to read {path}: {e}");
            std::process::exit(1);
        })
    }
}

/// Splices `section` right before the base template's
/// `<section class="methodology">` anchor (a literal, always-present
/// substring of `xval::report`'s `TEMPLATE` -- no dynamic content in that
/// opening tag). Falls back to appending at the end if the anchor is ever
/// missing (e.g. a future template rework), so the section is never
/// silently dropped.
fn inject_python_section(base_html: &str, section: &str) -> String {
    const ANCHOR: &str = r#"<section class="methodology">"#;
    match base_html.find(ANCHOR) {
        Some(pos) => {
            let mut out = String::with_capacity(base_html.len() + section.len());
            out.push_str(&base_html[..pos]);
            out.push_str(section);
            out.push_str(&base_html[pos..]);
            out
        }
        None => format!("{base_html}\n{section}"),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| {
        eprintln!("usage: render_report <report.json|-> [python_bench.json]");
        std::process::exit(2);
    });
    let python_path = args.next();

    let json = read_json_arg(&path);
    let base_html = match xval::render(&json) {
        Ok(html) => html,
        Err(e) => {
            eprintln!("render_report: {e}");
            std::process::exit(1);
        }
    };

    let html = match python_path {
        Some(py_path) => {
            let py_json = read_json_arg(&py_path);
            match xval::render_python(&py_json) {
                Ok(section) => inject_python_section(&base_html, &section),
                Err(e) => {
                    eprintln!("render_report: failed to render python bench section: {e}");
                    std::process::exit(1);
                }
            }
        }
        None => base_html,
    };

    print!("{html}");
}
