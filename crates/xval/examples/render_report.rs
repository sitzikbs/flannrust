//! CLI wrapper around `xval::render` -- turns the JSON emitted by
//! `examples/report_data.rs` into a self-contained HTML scorecard on
//! stdout. All the Rust-vs-C++ parsing/rendering logic lives in
//! `xval::report::render` (unit-tested in `tests/render_report_test.rs`
//! against a handcrafted JSON fixture); this binary is just argument
//! handling and I/O -- PLUS (M-py Task 5) an optional second "Python
//! bindings" section, rendered entirely in this file (not in the `xval`
//! library) from `crates/flannrust-py/python/bench/bench_py.py`'s emitted
//! JSON. That JSON is spliced into the base HTML right before the
//! `<section class="methodology">` anchor (see `inject_python_section`),
//! reusing the base template's existing CSS classes (`section`/`h2`/
//! `legend`/`pill`/`table`/`scroll-x`/`num`) -- no new stylesheet rules.
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

use serde::Deserialize;

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

// ============================================================================
// Python bench JSON shape -- mirrors `bench_py.py`'s emitted
// `{meta: {...}, workloads: [...]}` document. Deliberately NOT shared with
// `xval::report`'s `ReportDoc`/`Meta`/etc. (those are private to that
// module, and this file may only touch this example, not the library) --
// small, self-contained duplication of the parse/escape/format helpers
// below is the honest tradeoff, not a refactor opportunity.
// ============================================================================

#[derive(Debug, Deserialize)]
struct PyReport {
    meta: PyMeta,
    workloads: Vec<PyWorkload>,
}

#[derive(Debug, Deserialize)]
struct PyMeta {
    python: String,
    numpy: String,
    scipy: String,
    pynanoflann: String,
    cpu: String,
    threads: u64,
    date: String,
    git_sha: String,
    #[serde(default)]
    rustflags: String,
    #[serde(default)]
    wheel_profile: String,
}

#[derive(Debug, Deserialize)]
struct PyWorkload {
    name: String,
    flannrust_ms: f64,
    ckdtree_ms: f64,
    pynanoflann_ms: f64,
    ratio_ckdtree: f64,
    ratio_pynanoflann: f64,
    #[serde(default)]
    note: Option<String>,
}

/// Mirrors `xval::report`'s own (private) `html_escape` -- same five ASCII
/// characters, same reasoning (free-text from JSON, e.g. `note`/`cpu`,
/// must never be interpreted as markup).
fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// Adaptive-precision ms formatting: fixed `.3` decimals renders every
/// sub-millisecond value (e.g. the single-query-loop row's per-call times,
/// ~0.002 ms) as an indistinguishable "0.002" -- widen to 6 decimals below
/// 1.0 ms so distinct sub-ms values (e.g. 0.001968 vs 0.002273) stay
/// legible; every other row's values are >> 1 ms and render identically to
/// before (still `.3`).
fn fmt_ms(v: f64) -> String {
    if v.abs() < 1.0 {
        format!("{v:.6}")
    } else {
        format!("{v:.3}")
    }
}

/// Reuses the base template's existing `.pill` classes (`good`/`fail`) for
/// the ratio cells -- `ratio < 1.0` means flannrust is faster (per the
/// brief: "ratios < 1.0 = flannrust faster"), styled the same "good" green
/// as every other win-indicator on the page.
fn ratio_pill(ratio: f64) -> String {
    if ratio < 1.0 {
        format!(r#"<span class="pill good">{ratio:.3}&times; flannrust faster</span>"#)
    } else {
        format!(r#"<span class="pill fail">{ratio:.3}&times; other faster</span>"#)
    }
}

fn render_python_workload_rows(workloads: &[PyWorkload]) -> String {
    workloads
        .iter()
        .map(|w| {
            // Per the brief: "single-query row labeled 'overhead-bound (see
            // EXPERIMENTS)'".
            let overhead_badge = if w.name.contains("single_query") {
                r#" <span class="pill fail">overhead-bound (see EXPERIMENTS)</span>"#.to_string()
            } else {
                String::new()
            };
            let title_attr =
                w.note.as_deref().map(|n| format!(r#" title="{}""#, html_escape(n))).unwrap_or_default();

            format!(
                r#"<tr{title_attr}>
<td>{name}{overhead_badge}</td>
<td class="num">{f_ms}</td>
<td class="num">{c_ms}</td>
<td class="num">{p_ms}</td>
<td class="num">{c_ratio}</td>
<td class="num">{p_ratio}</td>
</tr>"#,
                title_attr = title_attr,
                name = html_escape(&w.name),
                overhead_badge = overhead_badge,
                f_ms = fmt_ms(w.flannrust_ms),
                c_ms = fmt_ms(w.ckdtree_ms),
                p_ms = fmt_ms(w.pynanoflann_ms),
                c_ratio = ratio_pill(w.ratio_ckdtree),
                p_ratio = ratio_pill(w.ratio_pynanoflann),
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parses `json` (the document `bench_py.py` emits) and renders a
/// self-contained `<section>` fragment -- same visual language as
/// `xval::report`'s sections (reuses `.legend`/`.pill`/`.table`/`.scroll-x`,
/// no new CSS), ready to splice into the base HTML.
fn render_python_section(json: &str) -> Result<String, String> {
    let doc: PyReport = serde_json::from_str(json).map_err(|e| format!("failed to parse python bench JSON: {e}"))?;

    let native = doc.meta.rustflags.contains("target-cpu=native");
    let meta_line = format!(
        "Python {} &middot; numpy {} &middot; scipy {} &middot; pynanoflann {} &middot; {} &middot; {} threads &middot; {} &middot; git {}{}",
        html_escape(&doc.meta.python),
        html_escape(&doc.meta.numpy),
        html_escape(&doc.meta.scipy),
        html_escape(&doc.meta.pynanoflann),
        html_escape(&doc.meta.cpu),
        doc.meta.threads,
        html_escape(&doc.meta.date),
        html_escape(&doc.meta.git_sha),
        if native { " &middot; target-cpu=native" } else { "" },
    );

    let rows = render_python_workload_rows(&doc.workloads);

    Ok(format!(
        r#"<section class="python-bindings">
<h2 title="{wheel_profile}">Python bindings vs cKDTree / pynanoflann</h2>
<p class="machine-line">{meta_line}</p>
<div class="legend"><span class="pill good">ratio &lt; 1.0</span>flannrust faster<span class="pill fail">ratio &ge; 1.0</span>other faster</div>
<div class="scroll-x">
<table>
<thead>
<tr><th>Workload</th><th>flannrust ms</th><th>cKDTree ms</th><th>pynanoflann ms</th><th>ratio vs cKDTree</th><th>ratio vs pynanoflann</th></tr>
</thead>
<tbody>
{rows}
</tbody>
</table>
</div>
</section>

"#,
        wheel_profile = html_escape(&doc.meta.wheel_profile),
        meta_line = meta_line,
        rows = rows,
    ))
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
            match render_python_section(&py_json) {
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
