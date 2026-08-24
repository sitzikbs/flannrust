//! HTML scorecard renderer for the JSON emitted by `examples/
//! report_data.rs` (see that file's module doc for the exact shape,
//! including M2 Task 5's dynamic-accuracy rows and this task's `meta`
//! extension). `render` is the entire public surface: parse -> compute a
//! few derived verdicts straight from the parsed numbers -> substitute
//! into a checked-in HTML template. NOTHING in the rendered page is a
//! hardcoded number -- every figure on the page traces back to a JSON
//! field read via `serde_json` (the one exception is genuinely static
//! page furniture: the title text, section headings, column labels).
//!
//! `examples/render_report.rs` is the thin CLI wrapper: `render_report
//! <report.json|->` writes the HTML to stdout.

use serde::Deserialize;
use std::fmt;

// ============================================================================
// Parsed JSON shape -- a deliberately loose mirror of report_data.rs's
// emitted document. Fields this renderer doesn't use are simply omitted
// from these structs (serde ignores unknown JSON fields by default, so a
// leaner-than-the-JSON struct is safe); every dynamic-row evidence field
// (`live_count`/`removed_count`/the `dyn_ops_stats` op-kind totals) is
// `Option` because only the dynamic accuracy rows carry it -- see
// `report_data.rs`'s `dynamic_accuracy_rows`.
// ============================================================================

#[derive(Debug, Deserialize)]
struct ReportDoc {
    meta: Meta,
    speed: Vec<SpeedRow>,
    accuracy: Vec<AccuracyRow>,
}

#[derive(Debug, Deserialize)]
struct Meta {
    date: String,
    nanoflann_version: String,
    rustc: String,
    cpu_threads: u64,
    target_cpu_native: bool,
    scoring: String,
    gt_methodology: String,
    speed_methodology: String,
    #[serde(default)]
    cpu_model: Option<String>,
    #[serde(default)]
    kernel: Option<String>,
    #[serde(default)]
    cxx_compiler: Option<String>,
    #[serde(default)]
    git_sha: Option<String>,
    #[serde(default)]
    wsl: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct SpeedRow {
    workload: String,
    rust_ms: f64,
    cpp_ms: f64,
    ratio: f64,
}

#[derive(Debug, Deserialize)]
struct AccuracyRow {
    workload: String,
    rust_exact_tie_aware_vs_bruteforce: f64,
    cpp_exact_tie_aware_vs_bruteforce: f64,
    rust_eq_cpp_bitexact: bool,
    mean_dist_rel_error_rust: f64,
    max_dist_rel_error_rust: f64,
    mean_dist_rel_error_cpp: f64,
    max_dist_rel_error_cpp: f64,
    #[serde(default)]
    live_count: Option<u64>,
    #[serde(default)]
    removed_count: Option<u64>,
    #[serde(default)]
    grow_and_add_count: Option<u64>,
    #[serde(default)]
    remove_count: Option<u64>,
    #[serde(default)]
    readd_count: Option<u64>,
    #[serde(default)]
    tombstone_migrations: Option<u64>,
}

/// Everything that can go wrong turning a `report_data` JSON document into
/// HTML -- currently just "the JSON didn't parse / didn't match the
/// expected shape". Kept as its own type (rather than exposing
/// `serde_json::Error` directly at the API boundary) so the public surface
/// doesn't leak a serde_json version-specific type.
#[derive(Debug)]
pub enum RenderError {
    Json(serde_json::Error),
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RenderError::Json(e) => write!(f, "failed to parse report JSON: {e}"),
        }
    }
}

impl std::error::Error for RenderError {}

impl From<serde_json::Error> for RenderError {
    fn from(e: serde_json::Error) -> Self {
        RenderError::Json(e)
    }
}

// ============================================================================
// HTML template -- a checked-in Rust string constant with `{placeholder}`
// slots, filled in via plain `str::replace` (NOT `format!`/`write!`'s
// `{}` syntax -- the CSS below is full of literal `{`/`}` braces that
// `format!` would otherwise demand be escaped as `{{`/`}}` throughout,
// which would make this template unreadable and easy to corrupt on edit).
//
// Design (Task 5b brief, validated):
// - IBM Plex Sans (body) + IBM Plex Mono (numerics) via Google Fonts, with
//   generic fallback stacks so the page still reads fine offline.
// - Three-state token theming: full LIGHT palette on bare `:root` (never
//   only inside a media/data-theme block -- that's the classic
//   broken-artifact bug where an explicit light choice or a media-query
//   miss leaves every token undefined), re-asserted for dark under
//   `@media (prefers-color-scheme: dark)` guarded by
//   `:root:not([data-theme="light"])` (so an explicit light override
//   still wins even in a dark OS), and AGAIN under `:root[data-theme="dark"]`
//   (so an explicit dark override wins regardless of OS preference).
//   `body`'s background comes from the `--bg` token, never a literal color,
//   so the viewer never sees a transparent/mismatched frame.
// - Palette: light bg #FAF8F5 / surface #FFF / ink #23201B / muted #6E675E /
//   line #E5E0D8 / rust #C8502E / cpp #2B6CB0 / good #2F855A / bad #B42318;
//   dark bg #1A1814 / surface #23201B / ink #EDE8E0 / muted #9C948A / line
//   #38332C / rust #D96A45 / cpp #4A8FC7 / good #58B287 / bad #F08578.
//   `--bad` (fix round 1) is its OWN dedicated token, deliberately distinct
//   from `--rust` -- aliasing it to `--rust` (the original design) meant
//   the same hue read as both "Rust faster" (bars/legend) and "failure"
//   (tiles/pills/caveat), a real cross-page misread risk. Failure state
//   also never relies on color alone: `render_tiles`/`render_accuracy_rows`
//   prefix every good/fail tile-value and pill with a `\u{2713} `/`\u{2715}
//   ` text glyph (see `GOOD_GLYPH`/`BAD_GLYPH` below), so the state reads
//   correctly even in grayscale/colorblind rendering.
const TEMPLATE: &str = r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>flannrust Scorecard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #FAF8F5;
  --surface: #FFFFFF;
  --ink: #23201B;
  --muted: #6E675E;
  --line: #E5E0D8;
  --rust: #C8502E;
  --cpp: #2B6CB0;
  --good: #2F855A;
  --bad: #B42318;
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #1A1814;
    --surface: #23201B;
    --ink: #EDE8E0;
    --muted: #9C948A;
    --line: #38332C;
    --rust: #D96A45;
    --cpp: #4A8FC7;
    --good: #58B287;
    --bad: #F08578;
  }
}

:root[data-theme="dark"] {
  --bg: #1A1814;
  --surface: #23201B;
  --ink: #EDE8E0;
  --muted: #9C948A;
  --line: #38332C;
  --rust: #D96A45;
  --cpp: #4A8FC7;
  --good: #58B287;
  --bad: #F08578;
}

* { box-sizing: border-box; }

body {
  background: var(--bg);
  color: var(--ink);
  margin: 0;
  font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  line-height: 1.5;
}

.mono, td.num, .bar-value, .tile-value {
  font-family: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums;
}

.page {
  max-width: 960px;
  margin: 0 auto;
  padding: 2.5rem 1.5rem 4rem;
}

header h1 {
  margin: 0 0 0.25rem;
  font-size: 1.75rem;
  font-weight: 700;
}

.machine-line {
  color: var(--muted);
  font-size: 0.9rem;
  margin: 0;
}

.caveat {
  margin: 0.75rem 0 0;
  padding: 0.6rem 0.9rem;
  border: 1px solid var(--bad);
  border-radius: 6px;
  color: var(--bad);
  background: color-mix(in srgb, var(--bad) 8%, var(--surface));
  font-size: 0.9rem;
}

.tiles {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1rem;
  margin: 2rem 0;
}

.tile {
  background: var(--surface);
  border: 1px solid var(--line);
  border-left: 4px solid var(--good);
  border-radius: 8px;
  padding: 1rem 1.1rem;
}

.tile.fail {
  border-left-color: var(--bad);
}

.tile-label {
  color: var(--muted);
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 0.35rem;
}

.tile-value {
  font-size: 1.15rem;
  font-weight: 600;
}

.tile.fail .tile-value { color: var(--bad); }
.tile.good .tile-value, .tile:not(.fail) .tile-value { color: var(--good); }

section { margin: 2.5rem 0; }

h2 {
  font-size: 1.15rem;
  border-bottom: 1px solid var(--line);
  padding-bottom: 0.4rem;
}

.legend {
  display: flex;
  gap: 1.5rem;
  align-items: center;
  color: var(--muted);
  font-size: 0.85rem;
  margin-bottom: 1rem;
}

.swatch {
  display: inline-block;
  width: 0.8rem;
  height: 0.8rem;
  border-radius: 2px;
  margin-right: 0.35rem;
  vertical-align: -0.1rem;
}

.swatch.rust { background: var(--rust); }
.swatch.cpp { background: var(--cpp); }

.scroll-x { overflow-x: auto; }

.bar-row {
  display: grid;
  grid-template-columns: 13rem 1fr 15rem;
  gap: 0.75rem;
  align-items: center;
  padding: 0.4rem 0;
  border-bottom: 1px solid var(--line);
  min-width: 640px;
}

.bar-label { font-size: 0.85rem; }

.track {
  position: relative;
  height: 1.1rem;
  background: var(--line);
  border-radius: 3px;
}

.parity-line {
  position: absolute;
  left: 50%;
  top: -3px;
  bottom: -3px;
  width: 2px;
  background: var(--muted);
}

.bar {
  position: absolute;
  top: 2px;
  bottom: 2px;
  border-radius: 3px;
  transition: filter 0.1s ease;
}

.bar.rust { background: var(--rust); }
.bar.cpp { background: var(--cpp); }
.bar:hover { filter: brightness(1.2); }

.bar-value { font-size: 0.82rem; color: var(--muted); }
.capped-note { color: var(--bad); margin-left: 0.35rem; }

table {
  border-collapse: collapse;
  width: 100%;
  min-width: 720px;
  font-size: 0.88rem;
}

th, td {
  text-align: left;
  padding: 0.5rem 0.7rem;
  border-bottom: 1px solid var(--line);
}

th {
  color: var(--muted);
  font-weight: 600;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}

td.good { color: var(--good); font-weight: 600; }

.pill {
  display: inline-block;
  padding: 0.15rem 0.55rem;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 600;
}

.pill.good { background: color-mix(in srgb, var(--good) 18%, var(--surface)); color: var(--good); }
.pill.fail { background: color-mix(in srgb, var(--bad) 18%, var(--surface)); color: var(--bad); }

.methodology p { color: var(--muted); font-size: 0.9rem; }
.methodology strong { color: var(--ink); }

footer {
  color: var(--muted);
  font-size: 0.8rem;
  border-top: 1px solid var(--line);
  padding-top: 1rem;
}
</style>
</head>
<body>
<div class="page">
<header>
<h1>flannrust Scorecard</h1>
<p class="machine-line">{date} &middot; {git_sha} &middot; {machine_line}</p>
{wsl_caveat}
</header>

{tiles}

<section class="speed">
<h2>Speed</h2>
<div class="legend"><span class="swatch rust"></span>Rust faster<span class="swatch cpp"></span>C++ faster</div>
<div class="scroll-x">
{speed_rows}
</div>
</section>

<section class="accuracy">
<h2>Accuracy vs brute-force ground truth</h2>
<div class="scroll-x">
<table>
<thead>
<tr><th>Workload</th><th>eps</th><th>Rust exact</th><th>C++ exact</th><th>Mean rel err (rust)</th><th>Max rel err (rust)</th><th>Mean rel err (cpp)</th><th>Max rel err (cpp)</th><th>Bit-exact</th></tr>
</thead>
<tbody>
{accuracy_rows}
</tbody>
</table>
</div>
</section>

<section class="methodology">
<h2>Methodology</h2>
<p><strong>Scoring:</strong> {scoring}</p>
<p><strong>Ground truth:</strong> {gt_methodology}</p>
<p><strong>Speed:</strong> {speed_methodology}</p>
</section>

<footer>
Generated by the xval suite &middot; {date} &middot; {git_sha}
</footer>
</div>
</body>
</html>
"##;

/// Escapes the five ASCII characters HTML gives special meaning, so
/// free-text pulled from JSON (methodology notes, compiler/kernel
/// strings, workload names) can never be interpreted as markup. Notably:
/// `meta.speed_methodology` contains literal `Vec<u32>`/`Vec<T>` --
/// without this, `<u32>` would be swallowed by the browser as an unknown
/// tag instead of displayed.
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

fn percent(x: f64) -> String {
    format!("{:.1}%", x * 100.0)
}

fn sci(x: f64) -> String {
    format!("{x:.2e}")
}

/// Fix round 1: text-glyph prefixes for good/fail states (tiles and
/// pills), so state reads correctly WITHOUT relying on `--good`/`--bad`
/// color alone (colorblind-safe / grayscale-safe). Plain text glyphs, no
/// icon font.
const GOOD_GLYPH: &str = "\u{2713} "; // "✓ "
const BAD_GLYPH: &str = "\u{2715} "; // "✕ "

/// Pulls the `eps<value>` suffix off a workload name (e.g.
/// `"uniform_dim3_f32_k10_eps0.1"` -> `Some("ε=0.1")`), matching
/// `report_data.rs`'s `EPS_VALUES`/`DYN_ACC_EPS_VALUES` naming convention
/// (`_eps0`, `_eps0.1`, `_eps1`). `None` if no such suffix is present.
fn parse_eps(workload: &str) -> Option<String> {
    workload.rfind("_eps").map(|i| format!("\u{3b5}={}", &workload[i + "_eps".len()..]))
}

/// Verdict tiles: (a) accuracy at eps=0 -- "100% exact @ eps=0" only if
/// EVERY eps=0 row (rust AND cpp) is exactly 1.0, else the actual minimum
/// across those rows, flagged as a failure; (b) bit-exactness -- only a
/// pass if EVERY accuracy row (any eps) has `rust_eq_cpp_bitexact`; (c)
/// best speed win -- `max(1/ratio)` across all speed rows, with that row's
/// workload name -- flagged as a failure (fix round 1) when that max is
/// still `< 1.0`, i.e. C++ was faster on every single speed row, so there
/// is genuinely no rust win to report. Every tile's value text is
/// glyph-prefixed (`GOOD_GLYPH`/`BAD_GLYPH`) in addition to its
/// good/fail CSS class, so the verdict reads without relying on color.
fn render_tiles(accuracy: &[AccuracyRow], speed: &[SpeedRow]) -> String {
    let eps0_rows: Vec<&AccuracyRow> = accuracy.iter().filter(|r| r.workload.ends_with("_eps0")).collect();
    let (acc_ok, acc_value) = if eps0_rows.is_empty() {
        (false, "no eps=0 rows in report".to_string())
    } else {
        let min = eps0_rows
            .iter()
            .flat_map(|r| [r.rust_exact_tie_aware_vs_bruteforce, r.cpp_exact_tie_aware_vs_bruteforce])
            .fold(f64::INFINITY, f64::min);
        if min >= 1.0 {
            (true, "100% exact @ eps=0".to_string())
        } else {
            (false, format!("{} exact @ eps=0 (worst row)", percent(min)))
        }
    };

    let total = accuracy.len();
    let exact_count = accuracy.iter().filter(|r| r.rust_eq_cpp_bitexact).count();
    let (bit_ok, bit_value) = if total > 0 && exact_count == total {
        (true, "Bit-exact vs C++ (all rows)".to_string())
    } else {
        (false, format!("{exact_count}/{total} rows bit-exact"))
    };

    let best = speed.iter().filter(|r| r.ratio > 0.0).map(|r| (1.0 / r.ratio, r.workload.as_str())).fold(
        (f64::NEG_INFINITY, ""),
        |acc, cur| if cur.0 > acc.0 { cur } else { acc },
    );
    let (speed_ok, speed_value) = if speed.is_empty() {
        (false, "no speed rows in report".to_string())
    } else if best.0 >= 1.0 {
        (true, format!("{:.2}\u{d7} faster \u{2014} {}", best.0, html_escape(best.1)))
    } else {
        // Every row was cpp-faster (`1/ratio < 1.0` everywhere) -- report
        // the least-bad row's actual ratio (`1/best.0`, since `best.0` is
        // `max(1/ratio)` == `1/min(ratio)`) honestly rather than dressing
        // up a loss as a "win".
        let min_ratio = 1.0 / best.0;
        (false, format!("no speed win \u{2014} best ratio {:.2}\u{d7} ({})", min_ratio, html_escape(best.1)))
    };

    format!(
        r#"<div class="tiles">
<div class="tile {}"><div class="tile-label">Accuracy @ eps=0</div><div class="tile-value">{}{}</div></div>
<div class="tile {}"><div class="tile-label">Bit-exactness</div><div class="tile-value">{}{}</div></div>
<div class="tile {}"><div class="tile-label">Best speed win</div><div class="tile-value">{}{}</div></div>
</div>"#,
        if acc_ok { "good" } else { "fail" },
        if acc_ok { GOOD_GLYPH } else { BAD_GLYPH },
        acc_value,
        if bit_ok { "good" } else { "fail" },
        if bit_ok { GOOD_GLYPH } else { BAD_GLYPH },
        bit_value,
        if speed_ok { "good" } else { "fail" },
        if speed_ok { GOOD_GLYPH } else { BAD_GLYPH },
        speed_value,
    )
}

/// Horizontal diverging bars around a parity midline: bar side/color by
/// who's faster (`ratio = rust_ms / cpp_ms`; `< 1.0` is rust-faster, `>
/// 1.0` is cpp-faster), width proportional to `|1 - ratio|` scaled so a
/// 70% difference (`ratio` 0.3 or 1.7) fills the full half-track (50%),
/// capped beyond that with a visible annotation (the numeric value column
/// always shows the true, uncapped ratio regardless).
fn render_speed_rows(speed: &[SpeedRow]) -> String {
    speed
        .iter()
        .map(|r| {
            let diff = (1.0 - r.ratio).abs();
            let capped = diff > 0.7;
            let pct = (diff / 0.7).min(1.0) * 50.0;
            let (side_class, style) = if r.ratio < 1.0 {
                ("rust", format!("width:{pct:.2}%; right:50%;"))
            } else if r.ratio > 1.0 {
                ("cpp", format!("width:{pct:.2}%; left:50%;"))
            } else {
                ("rust", "width:0%; right:50%;".to_string())
            };
            let cap_note =
                if capped { r#"<span class="capped-note">capped at &plusmn;70%</span>"# } else { "" };
            let tooltip = html_escape(&format!(
                "{}: ratio {:.3} (rust {:.3} ms / cpp {:.3} ms)",
                r.workload, r.ratio, r.rust_ms, r.cpp_ms
            ));
            format!(
                r#"<div class="bar-row" title="{tooltip}">
<div class="bar-label">{workload}</div>
<div class="track"><div class="parity-line"></div><div class="bar {side_class}" style="{style}"></div></div>
<div class="bar-value">{ratio:.3}&middot;{rust_ms:.3}&nbsp;ms&nbsp;/&nbsp;{cpp_ms:.3}&nbsp;ms{cap_note}</div>
</div>"#,
                tooltip = tooltip,
                workload = html_escape(&r.workload),
                side_class = side_class,
                style = style,
                ratio = r.ratio,
                rust_ms = r.rust_ms,
                cpp_ms = r.cpp_ms,
                cap_note = cap_note,
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Accuracy table rows: workload, eps parsed from the name, rust/cpp
/// exactness as a percentage (rendered in the `good` token when the row is
/// perfect on both sides), mean/max relative distance error in compact
/// scientific form, and a bit-exact pill. Dynamic-row evidence
/// (`live_count`/`removed_count`/op-kind totals), when present, is
/// surfaced as a native `title` tooltip on the row rather than as extra
/// visible columns (keeping the table's column set identical for static
/// and dynamic rows).
fn render_accuracy_rows(accuracy: &[AccuracyRow]) -> String {
    accuracy
        .iter()
        .map(|r| {
            let eps = parse_eps(&r.workload).unwrap_or_else(|| "\u{2014}".to_string());
            let perfect = r.rust_exact_tie_aware_vs_bruteforce >= 1.0 && r.cpp_exact_tie_aware_vs_bruteforce >= 1.0;
            let cell_class = if perfect { "good" } else { "" };
            let pill = if r.rust_eq_cpp_bitexact {
                format!(r#"<span class="pill good">{GOOD_GLYPH}bit-exact</span>"#)
            } else {
                format!(r#"<span class="pill fail">{BAD_GLYPH}not bit-exact</span>"#)
            };

            let evidence_attr = match (r.live_count, r.removed_count) {
                (Some(live), Some(removed)) => {
                    let ops = match (r.grow_and_add_count, r.remove_count, r.readd_count, r.tombstone_migrations) {
                        (Some(g), Some(rm), Some(ra), Some(tm)) => format!(
                            " grow_and_add={g} remove={rm} readd={ra} tombstone_migrations={tm}"
                        ),
                        _ => String::new(),
                    };
                    format!(r#" title="live={live} removed={removed}{ops}""#)
                }
                _ => String::new(),
            };

            format!(
                r#"<tr{evidence_attr}>
<td>{workload}</td>
<td class="num">{eps}</td>
<td class="num {cell_class}">{rust_pct}</td>
<td class="num {cell_class}">{cpp_pct}</td>
<td class="num">{mean_rust}</td>
<td class="num">{max_rust}</td>
<td class="num">{mean_cpp}</td>
<td class="num">{max_cpp}</td>
<td>{pill}</td>
</tr>"#,
                evidence_attr = evidence_attr,
                workload = html_escape(&r.workload),
                eps = html_escape(&eps),
                cell_class = cell_class,
                rust_pct = percent(r.rust_exact_tie_aware_vs_bruteforce),
                cpp_pct = percent(r.cpp_exact_tie_aware_vs_bruteforce),
                mean_rust = sci(r.mean_dist_rel_error_rust),
                max_rust = sci(r.max_dist_rel_error_rust),
                mean_cpp = sci(r.mean_dist_rel_error_cpp),
                max_cpp = sci(r.max_dist_rel_error_cpp),
                pill = pill,
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parses `json` (the document `examples/report_data.rs` emits) and
/// renders a complete, self-contained HTML scorecard: every number on the
/// page is read from `json`, nothing is hardcoded. Accepts a `path`
/// argument of `-` (handled by the `render_report` example, not here) to
/// mean "read from stdin".
pub fn render(json: &str) -> Result<String, RenderError> {
    let doc: ReportDoc = serde_json::from_str(json)?;

    let machine_line = format!(
        "{} &middot; {} threads &middot; {} &middot; rustc {} &middot; {} &middot; nanoflann {}{}",
        html_escape(doc.meta.cpu_model.as_deref().unwrap_or("unknown")),
        doc.meta.cpu_threads,
        html_escape(doc.meta.kernel.as_deref().unwrap_or("unknown")),
        html_escape(&doc.meta.rustc),
        html_escape(doc.meta.cxx_compiler.as_deref().unwrap_or("unknown")),
        html_escape(&doc.meta.nanoflann_version),
        if doc.meta.target_cpu_native { " &middot; target-cpu=native" } else { "" },
    );

    let wsl_caveat = if doc.meta.wsl.unwrap_or(false) {
        r#"<p class="caveat">Running under WSL: timings may include Windows-host scheduler/I-O noise; treat absolute numbers as indicative, not definitive.</p>"#.to_string()
    } else {
        String::new()
    };

    let html = TEMPLATE
        .replace("{date}", &html_escape(&doc.meta.date))
        .replace("{git_sha}", &html_escape(doc.meta.git_sha.as_deref().unwrap_or("unknown")))
        .replace("{machine_line}", &machine_line)
        .replace("{wsl_caveat}", &wsl_caveat)
        .replace("{tiles}", &render_tiles(&doc.accuracy, &doc.speed))
        .replace("{speed_rows}", &render_speed_rows(&doc.speed))
        .replace("{accuracy_rows}", &render_accuracy_rows(&doc.accuracy))
        .replace("{scoring}", &html_escape(&doc.meta.scoring))
        .replace("{gt_methodology}", &html_escape(&doc.meta.gt_methodology))
        .replace("{speed_methodology}", &html_escape(&doc.meta.speed_methodology));

    Ok(html)
}
