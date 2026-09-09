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

/// A deliberately loose subset of `xval::TimingStats` -- only the three
/// fields `render_speed_rows` actually displays (`mean_ms`/`std_ms`/`n`;
/// `median_ms`/`min_ms`/`max_ms` are present in the real JSON, emitted by
/// `report_data.rs`'s `timing_stats_json`, but simply ignored here since
/// `SpeedRow.rust_ms`/`cpp_ms` already carry the median -- see serde's
/// default "ignore unknown JSON fields" behavior). `Option`al on `SpeedRow`
/// (below) so a pre-M2.6 report.json (only `rust_ms`/`cpp_ms`/`ratio`, no
/// nested stats objects) still parses; `render_speed_rows` falls back to
/// the plain median-only display when absent.
#[derive(Debug, Deserialize)]
struct SpeedStats {
    mean_ms: f64,
    std_ms: f64,
    n: u64,
}

#[derive(Debug, Deserialize)]
struct SpeedRow {
    workload: String,
    /// Kept as the MEDIAN (per M2.6 -- see `report_data.rs`'s module doc):
    /// gate PASS/FAIL and the bar geometry below both key off medians, not
    /// means.
    rust_ms: f64,
    cpp_ms: f64,
    ratio: f64,
    #[serde(default)]
    rust: Option<SpeedStats>,
    #[serde(default)]
    cpp: Option<SpeedStats>,
    #[serde(default)]
    ratio_medians: Option<f64>,
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
    /// M2.6 task 3: `render_python`'s document carries a `schema_version`
    /// key specifically so an old-format `bench_py.py` JSON (pre-task-3:
    /// flat `flannrust_ms`/`ckdtree_ms`/`pynanoflann_ms` fields, no
    /// `schema_version` at all -- defaults to `0` below) is rejected with
    /// a clear, specific message INSTEAD OF whatever generic "missing
    /// field flannrust_stats" serde error would otherwise fire first --
    /// see `render_python`'s explicit version check, done before the full
    /// document is parsed.
    SchemaVersion {
        expected: u64,
        found: u64,
    },
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RenderError::Json(e) => write!(f, "failed to parse report JSON: {e}"),
            RenderError::SchemaVersion { expected, found } => write!(
                f,
                "python bench JSON schema_version mismatch: expected {expected}, found {found} -- \
                 re-run the updated bench_py.py to regenerate (an old-format JSON, without the \
                 mean/std/median/min/max/n stats objects this renderer expects, is no longer supported)"
            ),
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
    workload
        .rfind("_eps")
        .map(|i| format!("\u{3b5}={}", &workload[i + "_eps".len()..]))
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
    let eps0_rows: Vec<&AccuracyRow> = accuracy
        .iter()
        .filter(|r| r.workload.ends_with("_eps0"))
        .collect();
    let (acc_ok, acc_value) = if eps0_rows.is_empty() {
        (false, "no eps=0 rows in report".to_string())
    } else {
        let min = eps0_rows
            .iter()
            .flat_map(|r| {
                [
                    r.rust_exact_tie_aware_vs_bruteforce,
                    r.cpp_exact_tie_aware_vs_bruteforce,
                ]
            })
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

    let best = speed
        .iter()
        .filter(|r| r.ratio > 0.0)
        .map(|r| (1.0 / r.ratio, r.workload.as_str()))
        .fold((f64::NEG_INFINITY, ""), |acc, cur| {
            if cur.0 > acc.0 {
                cur
            } else {
                acc
            }
        });
    let (speed_ok, speed_value) = if speed.is_empty() {
        (false, "no speed rows in report".to_string())
    } else if best.0 >= 1.0 {
        (
            true,
            format!(
                "{:.2}\u{d7} faster \u{2014} {}",
                best.0,
                html_escape(best.1)
            ),
        )
    } else {
        // Every row was cpp-faster (`1/ratio < 1.0` everywhere) -- report
        // the least-bad row's actual ratio (`1/best.0`, since `best.0` is
        // `max(1/ratio)` == `1/min(ratio)`) honestly rather than dressing
        // up a loss as a "win".
        let min_ratio = 1.0 / best.0;
        (
            false,
            format!(
                "no speed win \u{2014} best ratio {:.2}\u{d7} ({})",
                min_ratio,
                html_escape(best.1)
            ),
        )
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
/// who's faster (`ratio` -- `< 1.0` is rust-faster, `> 1.0` is cpp-faster;
/// M2.6: this is `ratio_medians` when the row carries it, else the legacy
/// `ratio` field -- see `render_speed_rows`'s bar-geometry comment), width
/// proportional to `|1 - ratio|` scaled so a 70% difference (`ratio` 0.3 or
/// 1.7) fills the full half-track (50%), capped beyond that with a visible
/// annotation (the numeric value column always shows the true, uncapped
/// ratio regardless).
///
/// M2.6: when a row carries the new `rust`/`cpp` `TimingStats` objects, the
/// tooltip and bar-value cell additionally show `mean &plusmn; std (n=N)`
/// for BOTH sides -- the bar's geometry itself still keys off the ratio of
/// MEDIANS (gate decisions and doc ratio claims are median-based, not
/// mean-based -- see `report_data.rs`'s module doc). A pre-M2.6 row
/// (`rust`/`cpp` both `None`) falls back to the original plain
/// `ratio (rust ms / cpp ms)` display unchanged.
fn render_speed_rows(speed: &[SpeedRow]) -> String {
    speed
        .iter()
        .map(|r| {
            let ratio = r.ratio_medians.unwrap_or(r.ratio);
            let diff = (1.0 - ratio).abs();
            let capped = diff > 0.7;
            let pct = (diff / 0.7).min(1.0) * 50.0;
            let (side_class, style) = if ratio < 1.0 {
                ("rust", format!("width:{pct:.2}%; right:50%;"))
            } else if ratio > 1.0 {
                ("cpp", format!("width:{pct:.2}%; left:50%;"))
            } else {
                ("rust", "width:0%; right:50%;".to_string())
            };
            let cap_note =
                if capped { r#"<span class="capped-note">capped at &plusmn;70%</span>"# } else { "" };
            // Raw HTML (goes directly into the bar-value `<div>`, not
            // through `html_escape`) -- entities, matching the rest of
            // this function's markup.
            let stats_suffix = match (&r.rust, &r.cpp) {
                (Some(rs), Some(cs)) => format!(
                    " | rust {:.3}&plusmn;{:.3}&nbsp;ms&nbsp;(n={}) &middot; cpp {:.3}&plusmn;{:.3}&nbsp;ms&nbsp;(n={})",
                    rs.mean_ms, rs.std_ms, rs.n, cs.mean_ms, cs.std_ms, cs.n
                ),
                _ => String::new(),
            };
            // Plain Unicode (this text is built THEN passed through
            // `html_escape` below, which escapes `&` -- an HTML entity
            // string here would come out as literal "&plusmn;" text, not
            // the "\u{b1}" glyph, so this variant uses the raw characters
            // directly instead of reusing `stats_suffix`).
            let tooltip_stats_suffix = match (&r.rust, &r.cpp) {
                (Some(rs), Some(cs)) => format!(
                    " | rust {:.3}\u{b1}{:.3} ms (n={}) \u{b7} cpp {:.3}\u{b1}{:.3} ms (n={})",
                    rs.mean_ms, rs.std_ms, rs.n, cs.mean_ms, cs.std_ms, cs.n
                ),
                _ => String::new(),
            };
            let tooltip = html_escape(&format!(
                "{}: ratio_medians {:.3} (rust median {:.3} ms / cpp median {:.3} ms){}",
                r.workload, ratio, r.rust_ms, r.cpp_ms, tooltip_stats_suffix
            ));
            format!(
                r#"<div class="bar-row" title="{tooltip}">
<div class="bar-label">{workload}</div>
<div class="track"><div class="parity-line"></div><div class="bar {side_class}" style="{style}"></div></div>
<div class="bar-value">{ratio:.3}&middot;{rust_ms:.3}&nbsp;ms&nbsp;/&nbsp;{cpp_ms:.3}&nbsp;ms{cap_note}{stats_suffix}</div>
</div>"#,
                tooltip = tooltip,
                workload = html_escape(&r.workload),
                side_class = side_class,
                style = style,
                ratio = ratio,
                rust_ms = r.rust_ms,
                cpp_ms = r.cpp_ms,
                cap_note = cap_note,
                stats_suffix = stats_suffix,
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
            let perfect = r.rust_exact_tie_aware_vs_bruteforce >= 1.0
                && r.cpp_exact_tie_aware_vs_bruteforce >= 1.0;
            let cell_class = if perfect { "good" } else { "" };
            let pill = if r.rust_eq_cpp_bitexact {
                format!(r#"<span class="pill good">{GOOD_GLYPH}bit-exact</span>"#)
            } else {
                format!(r#"<span class="pill fail">{BAD_GLYPH}not bit-exact</span>"#)
            };

            let evidence_attr = match (r.live_count, r.removed_count) {
                (Some(live), Some(removed)) => {
                    let ops = match (
                        r.grow_and_add_count,
                        r.remove_count,
                        r.readd_count,
                        r.tombstone_migrations,
                    ) {
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

    let machine_line =
        format!(
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
        .replace(
            "{git_sha}",
            &html_escape(doc.meta.git_sha.as_deref().unwrap_or("unknown")),
        )
        .replace("{machine_line}", &machine_line)
        .replace("{wsl_caveat}", &wsl_caveat)
        .replace("{tiles}", &render_tiles(&doc.accuracy, &doc.speed))
        .replace("{speed_rows}", &render_speed_rows(&doc.speed))
        .replace("{accuracy_rows}", &render_accuracy_rows(&doc.accuracy))
        .replace("{scoring}", &html_escape(&doc.meta.scoring))
        .replace("{gt_methodology}", &html_escape(&doc.meta.gt_methodology))
        .replace(
            "{speed_methodology}",
            &html_escape(&doc.meta.speed_methodology),
        );

    Ok(html)
}

// ============================================================================
// M2.6 task 3 -- "Python bindings" section, rendered from
// `crates/flannrust-py/python/bench/bench_py.py`'s emitted JSON. Lives in
// the library (not `examples/render_report.rs`, where the pre-task-3
// version of this code lived) specifically so `tests/render_report_test.rs`
// can unit-test it directly, the same way `render` above is tested --
// `render_report.rs` is now just the thin CLI/stdin-vs-file/splice wrapper
// around this function. Deliberately its own small parse/escape/format
// surface (reuses `html_escape` above, but not `ReportDoc`/`SpeedRow`/etc.
// -- a different JSON document entirely, from a different producer).
// ============================================================================

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

/// One engine's adaptive-repetition stats for one workload row --
/// `bench_py.py`'s `stats_from_samples` mirrors `xval::TimingStats`
/// field-for-field; `min_ms`/`max_ms` are present in the real JSON but
/// unused by this renderer (serde silently ignores unknown/unread JSON
/// fields), same "deliberately loose subset" tradeoff as `report.rs`'s own
/// `SpeedStats`.
#[derive(Debug, Deserialize)]
struct PyStats {
    mean_ms: f64,
    std_ms: f64,
    median_ms: f64,
    n: u64,
}

#[derive(Debug, Deserialize)]
struct PyWorkload {
    name: String,
    flannrust_stats: PyStats,
    ckdtree_stats: PyStats,
    pynanoflann_stats: PyStats,
    /// Median-based (per M2.6 -- same decision statistic as the Rust-vs-C++
    /// section's `ratio_medians`), computed by `bench_py.py` itself.
    ratio_ckdtree: f64,
    ratio_pynanoflann: f64,
    #[serde(default)]
    note: Option<String>,
}

/// Only used to peek at `schema_version` BEFORE attempting the full
/// `PyReport` parse -- see `render_python`'s ordering comment. `#[serde(default)]`
/// so a document that omits the key entirely (every pre-task-3 JSON) reads
/// as `0`, which never matches `EXPECTED_SCHEMA_VERSION` below, rather than
/// failing to parse this tiny probe struct itself.
#[derive(Debug, Deserialize)]
struct PySchemaCheck {
    #[serde(default)]
    schema_version: u64,
}

#[derive(Debug, Deserialize)]
struct PyReport {
    meta: PyMeta,
    workloads: Vec<PyWorkload>,
}

const EXPECTED_PY_SCHEMA_VERSION: u64 = 2;

/// Reuses the base template's existing `.pill` classes (`good`/`fail`) for
/// the ratio cells -- `ratio < 1.0` means flannrust is faster (per the
/// brief: "ratios < 1.0 = flannrust faster"), styled the same "good" green
/// as every other win-indicator on the page. `ratio` here is always
/// `bench_py.py`'s median-based ratio (the only kind this schema emits).
fn ratio_pill_py(ratio: f64) -> String {
    if ratio < 1.0 {
        format!(r#"<span class="pill good">{ratio:.3}&times; flannrust faster</span>"#)
    } else {
        format!(r#"<span class="pill fail">{ratio:.3}&times; other faster</span>"#)
    }
}

/// `median_ms` as the headline value (same statistic the ratio pill uses),
/// with `mean &plusmn; std (n=N)` alongside -- mirrors `render_speed_rows`'s
/// `stats_suffix`/tooltip pattern above, adapted to this section's
/// fixed-column (not diverging-bar) table layout. Deliberately no new CSS
/// class (reuses only the existing `.num` cell styling) -- plain inline
/// text, same as `render_speed_rows`'s `stats_suffix`.
///
/// Precision is chosen ONCE from `median_ms`'s own magnitude (not
/// independently per number) -- `fmt_ms`'s per-value adaptive precision
/// would otherwise print e.g. `10.295 (10.338±0.250652, n=100)` (std < 1.0
/// ms widened to 6 decimals while the median/mean next to it stay at 3),
/// which is legible but visually inconsistent within one cell; a std
/// smaller than its own median is the common case, not the exception.
fn fmt_py_stat_cell(s: &PyStats) -> String {
    let (median, mean, std) = if s.median_ms.abs() < 1.0 {
        (
            format!("{:.6}", s.median_ms),
            format!("{:.6}", s.mean_ms),
            format!("{:.6}", s.std_ms),
        )
    } else {
        (
            format!("{:.3}", s.median_ms),
            format!("{:.3}", s.mean_ms),
            format!("{:.3}", s.std_ms),
        )
    };
    format!("{median}&nbsp;({mean}&plusmn;{std},&nbsp;n={n})", n = s.n)
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
            let tooltip = format!(
                "{name}: rust {rmean:.3}\u{b1}{rstd:.3} ms (n={rn}) \u{b7} cKDTree {cmean:.3}\u{b1}{cstd:.3} ms \
                 (n={cn}) \u{b7} pynanoflann {pmean:.3}\u{b1}{pstd:.3} ms (n={pn}){note}",
                name = w.name,
                rmean = w.flannrust_stats.mean_ms,
                rstd = w.flannrust_stats.std_ms,
                rn = w.flannrust_stats.n,
                cmean = w.ckdtree_stats.mean_ms,
                cstd = w.ckdtree_stats.std_ms,
                cn = w.ckdtree_stats.n,
                pmean = w.pynanoflann_stats.mean_ms,
                pstd = w.pynanoflann_stats.std_ms,
                pn = w.pynanoflann_stats.n,
                note = w.note.as_deref().map(|n| format!(" \u{b7} {n}")).unwrap_or_default(),
            );
            let title_attr = format!(r#" title="{}""#, html_escape(&tooltip));

            format!(
                r#"<tr{title_attr}>
<td>{name}{overhead_badge}</td>
<td class="num">{f_cell}</td>
<td class="num">{c_cell}</td>
<td class="num">{p_cell}</td>
<td class="num">{c_ratio}</td>
<td class="num">{p_ratio}</td>
</tr>"#,
                title_attr = title_attr,
                name = html_escape(&w.name),
                overhead_badge = overhead_badge,
                f_cell = fmt_py_stat_cell(&w.flannrust_stats),
                c_cell = fmt_py_stat_cell(&w.ckdtree_stats),
                p_cell = fmt_py_stat_cell(&w.pynanoflann_stats),
                c_ratio = ratio_pill_py(w.ratio_ckdtree),
                p_ratio = ratio_pill_py(w.ratio_pynanoflann),
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parses `json` (the document `bench_py.py` emits) and renders a
/// self-contained `<section>` fragment -- same visual language as this
/// module's own template (reuses `.legend`/`.pill`/`.table`/`.scroll-x`,
/// no new CSS), ready to splice into the base HTML by
/// `examples/render_report.rs`'s `inject_python_section`.
///
/// The `schema_version` check runs FIRST, via the tiny `PySchemaCheck`
/// probe struct, deliberately BEFORE the full `PyReport` parse -- an
/// old-format document (no `schema_version` key, flat `flannrust_ms`
/// instead of nested `flannrust_stats`) would also fail the full parse
/// with a generic "missing field" serde error, but that error doesn't say
/// WHY (a schema mismatch, not a typo/corruption) or what to do about it
/// (re-run bench_py.py) -- see `RenderError::SchemaVersion`'s doc comment.
pub fn render_python(json: &str) -> Result<String, RenderError> {
    let check: PySchemaCheck = serde_json::from_str(json)?;
    if check.schema_version != EXPECTED_PY_SCHEMA_VERSION {
        return Err(RenderError::SchemaVersion {
            expected: EXPECTED_PY_SCHEMA_VERSION,
            found: check.schema_version,
        });
    }

    let doc: PyReport = serde_json::from_str(json)?;

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
<tr><th>Workload</th><th>flannrust ms (median / mean&plusmn;std, n)</th><th>cKDTree ms</th><th>pynanoflann ms</th><th>ratio vs cKDTree (medians)</th><th>ratio vs pynanoflann (medians)</th></tr>
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
