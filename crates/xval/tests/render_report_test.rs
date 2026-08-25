//! TDD RED-first coverage for `xval::render` (the `report_data` JSON ->
//! self-contained HTML scorecard renderer backing `examples/
//! render_report.rs`). Feeds a small HANDCRAFTED JSON document (2 speed
//! rows -- one rust-faster, one cpp-faster; 2 accuracy rows -- one perfect
//! eps=0 row, one imperfect eps=0.1 row -- plus `meta.wsl: true`) through
//! `render` and asserts the rendered HTML actually reflects that data:
//! the parity-line markup, both bar-side classes, the tile values computed
//! FROM this exact JSON (not hardcoded), the WSL caveat text, and no
//! leftover `{placeholder}` template residue.
//!
//! Fix round 1: `--bad` is now its own dedicated color token (no longer an
//! alias of `--rust` -- the original design meant the same hue read as
//! both "Rust faster" and "failure", a real cross-page misread risk for a
//! public scorecard), and every good/fail tile-value and pill is prefixed
//! with a text glyph (`GOOD_GLYPH`/`BAD_GLYPH` below, mirroring `xval::
//! report`'s own constants) so state reads without relying on color at
//! all. Also: the "best speed win" tile is now honest -- if C++ was
//! faster on EVERY speed row (`max(1/ratio) < 1.0`), the tile reports "no
//! speed win" (fail-styled) instead of dressing up a loss as a win; see
//! `render_flags_best_speed_win_tile_as_failure_when_cpp_wins_everywhere`
//! (RED-captured against the pre-fix-round-1 renderer before the fix was
//! applied -- see task-5b-report.md's fix-round-1 addendum).

/// Mirrors `xval::report`'s own (private) `GOOD_GLYPH`/`BAD_GLYPH`
/// constants -- kept in lockstep by hand since this is a black-box test
/// crate with no access to the private constants themselves.
const GOOD_GLYPH: &str = "\u{2713} "; // "✓ "
const BAD_GLYPH: &str = "\u{2715} "; // "✕ "

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
    assert!(html.contains("<title>flannrust Scorecard</title>"), "missing/wrong <title>");
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
    // tile must read the all-exact success string, not a hardcoded pass/fail,
    // and (fix round 1) must be GOOD_GLYPH-prefixed, not just color-coded.
    let expected = format!("{GOOD_GLYPH}100% exact @ eps=0");
    assert!(html.contains(&expected), "accuracy tile did not report the glyph-prefixed all-exact verdict");
}

#[test]
fn render_computes_bitexact_tile_from_data() {
    let html = xval::render(FIXTURE).expect("render should succeed");
    // Fixture has 2 accuracy rows, only 1 bit-exact -- must NOT claim full
    // bit-exactness, and (fix round 1) must be BAD_GLYPH-prefixed.
    assert!(!html.contains("Bit-exact vs C++ (all rows)"), "bitexact tile wrongly claims full bit-exactness");
    let expected = format!("{BAD_GLYPH}1/2 rows bit-exact");
    assert!(html.contains(&expected), "bitexact tile did not report the glyph-prefixed 1/2 count");
}

#[test]
fn render_computes_best_speed_win_tile_from_data() {
    let html = xval::render(FIXTURE).expect("render should succeed");
    // max(1/ratio) across the two speed rows: 1/0.5 = 2.0 for
    // build_100k_dim3_f32_seq (the other row's 1/2.0 = 0.5 loses). This is
    // a genuine rust win (>= 1.0), so (fix round 1) it must be
    // GOOD_GLYPH-prefixed and use the "good" tile class.
    let expected = format!("{GOOD_GLYPH}2.00\u{d7} faster \u{2014} build_100k_dim3_f32_seq");
    assert!(html.contains(&expected), "best speed win tile did not report the glyph-prefixed computed factor");
    assert!(
        html.contains(r#"<div class="tile good"><div class="tile-label">Best speed win</div>"#),
        "best speed win tile must use the good class for a genuine rust win"
    );
}

#[test]
fn render_pills_are_glyph_prefixed() {
    let html = xval::render(FIXTURE).expect("render should succeed");
    // Fixture's eps0 row is bit-exact, eps0.1 row is not -- both pills must
    // carry their glyph, not just their color class.
    assert!(
        html.contains(&format!(r#"<span class="pill good">{GOOD_GLYPH}bit-exact</span>"#)),
        "bit-exact pill missing GOOD_GLYPH prefix"
    );
    assert!(
        html.contains(&format!(r#"<span class="pill fail">{BAD_GLYPH}not bit-exact</span>"#)),
        "not-bit-exact pill missing BAD_GLYPH prefix"
    );
}

/// All speed rows have C++ faster (`ratio > 1.0` on both), so
/// `max(1/ratio) < 1.0` -- there is genuinely no rust speed win to report
/// anywhere in this document.
const ALL_CPP_FASTER_FIXTURE: &str = r#"{
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
    "git_sha": "abc1234",
    "wsl": false,
    "scoring": "SCORING NOTE TEXT",
    "gt_methodology": "GT NOTE TEXT",
    "speed_methodology": "SPEED NOTE TEXT"
  },
  "speed": [
    { "workload": "build_100k_dim3_f32_seq", "rust_ms": 12.000000, "cpp_ms": 10.000000, "ratio": 1.200000 },
    { "workload": "knn_dim3_f32_k10", "rust_ms": 15.000000, "cpp_ms": 10.000000, "ratio": 1.500000 }
  ],
  "accuracy": [
    { "workload": "uniform_dim3_f32_k10_eps0", "n_queries": 2000, "rust_exact_tie_aware_vs_bruteforce": 1.000000, "cpp_exact_tie_aware_vs_bruteforce": 1.000000, "rust_eq_cpp_bitexact": true, "mean_dist_rel_error_rust": 0.000000, "max_dist_rel_error_rust": 0.000000, "mean_dist_rel_error_cpp": 0.000000, "max_dist_rel_error_cpp": 0.000000 }
  ]
}"#;

#[test]
fn render_flags_best_speed_win_tile_as_failure_when_cpp_wins_everywhere() {
    let html = xval::render(ALL_CPP_FASTER_FIXTURE).expect("render should succeed");

    // Must NOT dress up a loss as a win: no "X.XX× faster" claim anywhere,
    // and no "good" class on the best-speed-win tile.
    assert!(!html.contains("faster \u{2014}"), "tile must not claim a speed win when cpp wins every row");
    assert!(
        !html.contains(r#"<div class="tile good"><div class="tile-label">Best speed win</div>"#),
        "best speed win tile must not use the good class when there is no rust win"
    );

    // Least-bad row is build_100k_dim3_f32_seq at ratio 1.20 (the minimum
    // ratio across both rows, i.e. `1 / max(1/ratio)`) -- reported
    // honestly, fail-styled, BAD_GLYPH-prefixed.
    let expected = format!("{BAD_GLYPH}no speed win \u{2014} best ratio 1.20\u{d7} (build_100k_dim3_f32_seq)");
    assert!(
        html.contains(&format!(
            r#"<div class="tile fail"><div class="tile-label">Best speed win</div><div class="tile-value">{expected}</div></div>"#
        )),
        "best speed win tile did not report the honest no-win verdict:\n\n{html}"
    );
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
