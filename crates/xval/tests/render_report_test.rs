//! TDD RED-first coverage for `xval::render` (the `report_data` JSON ->
//! self-contained HTML scorecard renderer backing `examples/
//! render_report.rs`). Feeds a small HANDCRAFTED JSON document (2 speed
//! rows -- one rust-faster, one cpp-faster; 2 accuracy rows -- one perfect
//! eps=0 row, one imperfect eps=0.1 row -- plus `meta.wsl: true`) through
//! `render` and asserts the rendered HTML actually reflects that data:
//! the parity-line markup, both bar-side classes, the tile values computed
//! FROM this exact JSON (not hardcoded), the WSL caveat text, and no
//! leftover `{placeholder}` template residue.

const FIXTURE: &str = r#"{
  "meta": {
    "date": "2026-08-22T12:00:00Z",
    "nanoflann_version": "1.12.1",
    "rustc": "rustc 1.98.0 (88d9e12ae 2026-08-18)",
    "cpu_threads": 8,
    "target_cpu_native": true,
    "m2_dynamic": true,
    "cpu_model": "Test CPU Model X9",
    "kernel": "Linux 6.6.87.2-microsoft-standard-WSL2",
    "cxx_compiler": "g++ (GCC) 12.2.0",
    "git_sha": "abc1234-dirty",
    "wsl": true,
    "scoring": "SCORING NOTE TEXT with <angle> brackets & \"quotes\"",
    "gt_methodology": "GT NOTE TEXT",
    "speed_methodology": "SPEED NOTE TEXT with Vec<u32>/Vec<T>"
  },
  "speed": [
    { "workload": "build_100k_dim3_f32_seq", "rust_ms": 10.000000, "cpp_ms": 20.000000, "ratio": 0.500000 },
    { "workload": "knn_dim3_f32_k10", "rust_ms": 30.000000, "cpp_ms": 15.000000, "ratio": 2.000000 }
  ],
  "accuracy": [
    { "workload": "uniform_dim3_f32_k10_eps0", "n_queries": 2000, "rust_exact_tie_aware_vs_bruteforce": 1.000000, "cpp_exact_tie_aware_vs_bruteforce": 1.000000, "rust_eq_cpp_bitexact": true, "mean_dist_rel_error_rust": 0.000000, "max_dist_rel_error_rust": 0.000000, "mean_dist_rel_error_cpp": 0.000000, "max_dist_rel_error_cpp": 0.000000 },
    { "workload": "uniform_dim3_f32_k10_eps0.1", "n_queries": 2000, "rust_exact_tie_aware_vs_bruteforce": 0.870000, "cpp_exact_tie_aware_vs_bruteforce": 0.910000, "rust_eq_cpp_bitexact": false, "mean_dist_rel_error_rust": 0.000100, "max_dist_rel_error_rust": 0.002000, "mean_dist_rel_error_cpp": 0.000080, "max_dist_rel_error_cpp": 0.001800 }
  ]
}"#;

/// The template's own placeholder tokens (see `xval::report`'s `TEMPLATE`
/// constant) -- every one of these must be gone from real output; their
/// presence would mean a `.replace()` call was missed or misnamed.
const PLACEHOLDER_TOKENS: &[&str] = &[
    "{date}",
    "{git_sha}",
    "{machine_line}",
    "{wsl_caveat}",
    "{tiles}",
    "{speed_rows}",
    "{accuracy_rows}",
    "{scoring}",
    "{gt_methodology}",
    "{speed_methodology}",
];

#[test]
fn render_produces_html_with_no_placeholder_residue() {
    let html = xval::render(FIXTURE).expect("render should succeed on well-formed JSON");

    for token in PLACEHOLDER_TOKENS {
        assert!(!html.contains(token), "leftover template placeholder residue: {token}\n\n{html}");
    }
}

#[test]
fn render_includes_title_and_doctype() {
    let html = xval::render(FIXTURE).expect("render should succeed");
    assert!(html.contains("<!doctype html>") || html.contains("<!DOCTYPE html>"), "missing doctype");
    assert!(html.contains("<title>nanoflann-rs Scorecard</title>"), "missing/wrong <title>");
}

#[test]
fn render_includes_parity_line_markup() {
    let html = xval::render(FIXTURE).expect("render should succeed");
    assert!(html.contains("parity-line"), "missing the diverging-bar parity midline markup");
}

#[test]
fn render_includes_both_bar_side_classes() {
    let html = xval::render(FIXTURE).expect("render should succeed");
    // Fixture row 1 (ratio 0.5 < 1) is rust-faster; row 2 (ratio 2.0 > 1) is
    // cpp-faster -- both bar-side classes must appear in the rendered bars.
    assert!(html.contains(r#"class="bar rust""#), "missing rust-faster bar class");
    assert!(html.contains(r#"class="bar cpp""#), "missing cpp-faster bar class");
}

#[test]
fn render_computes_accuracy_tile_from_data() {
    let html = xval::render(FIXTURE).expect("render should succeed");
    // The only eps=0 row in the fixture is perfect (both sides 1.0), so the
    // tile must read the all-exact success string, not a hardcoded pass/fail.
    assert!(html.contains("100% exact @ eps=0"), "accuracy tile did not report the computed all-exact verdict");
}

#[test]
fn render_computes_bitexact_tile_from_data() {
    let html = xval::render(FIXTURE).expect("render should succeed");
    // Fixture has 2 accuracy rows, only 1 bit-exact -- must NOT claim full
    // bit-exactness.
    assert!(!html.contains("Bit-exact vs C++ (all rows)"), "bitexact tile wrongly claims full bit-exactness");
    assert!(html.contains("1/2"), "bitexact tile did not report the computed 1/2 count");
}

#[test]
fn render_computes_best_speed_win_tile_from_data() {
    let html = xval::render(FIXTURE).expect("render should succeed");
    // max(1/ratio) across the two speed rows: 1/0.5 = 2.0 for
    // build_100k_dim3_f32_seq (the other row's 1/2.0 = 0.5 loses).
    assert!(html.contains("2.00"), "best speed win tile did not report the computed 2.00x factor");
    assert!(html.contains("build_100k_dim3_f32_seq"), "best speed win tile did not name the winning workload");
}

#[test]
fn render_includes_wsl_caveat_when_set() {
    let html = xval::render(FIXTURE).expect("render should succeed");
    assert!(html.to_lowercase().contains("wsl"), "meta.wsl=true must surface a visible WSL-noise caveat");
}

#[test]
fn render_omits_wsl_caveat_when_unset() {
    let fixture_no_wsl = FIXTURE.replace(r#""wsl": true"#, r#""wsl": false"#);
    let html = xval::render(&fixture_no_wsl).expect("render should succeed");
    // The caveat PARAGRAPH itself must be gone -- note the stylesheet's
    // `.caveat` CSS rule is always present (static page furniture), so
    // this checks for the actual rendered element, not the class name.
    assert!(!html.contains(r#"<p class="caveat">"#), "wsl=false must not render the WSL-noise caveat paragraph");
}

#[test]
fn render_html_escapes_methodology_text() {
    let html = xval::render(FIXTURE).expect("render should succeed");
    // meta.scoring / meta.speed_methodology in the fixture deliberately
    // contain `<`, `>`, `&`, `"` -- these must be escaped, not passed
    // through raw (raw `<u32>` would be swallowed as an unknown tag).
    assert!(!html.contains("Vec<u32>"), "unescaped `<`/`>` from meta.speed_methodology leaked into HTML");
    assert!(html.contains("Vec&lt;u32&gt;"), "expected escaped Vec&lt;u32&gt; in methodology text");
}

#[test]
fn render_rejects_malformed_json() {
    let err = xval::render("{ not json").expect_err("malformed JSON must not render");
    let msg = err.to_string();
    assert!(!msg.is_empty(), "render error should carry a message");
}
