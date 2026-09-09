# Task 7 report: blog post for itzikbs.com

## Status: DONE

Branch `flannrust-post` in ~/dev/my_website (off `main`, main untouched), commit `61143ee`.

## Files created

- `blog/posts-md/2026-08-30-flannrust-ai-agent-port.md` — the post (~2,100 words)
- `assets/images/blog/flannrust-ratios.svg` — headline chart: 8 core workloads, Rust/C++ ratio of medians, noise band 0.94-1.16 shaded, wider fresh envelope 0.884-1.176 disclosed in subtitle
- `assets/images/blog/flannrust-dynadd.svg` — dyn_add before/after fidelity fix (1.133-1.144 vs 1.034-1.038, deviation-from-parity bars with session-range whiskers)
- `assets/images/blog/flannrust-python.svg` — cKDTree + pynanoflann median time relative to flannrust, 5 workloads, incl. the honest 0.84x pynanoflann win on dim8-f64

## Process compliance

- Humanizer SKILL.md read in full before writing; final pass verified: no em/en dashes in prose, no curly quotes, no AI-tell openers, "honest" trimmed from 6 to 4 uses, no bold-label lists, sentence-case section headings, one triad kept deliberately for rhythm.
- dataviz skill loaded before charts. Reference palette used verbatim (slots 1-2 blue/orange, light surface #fcfcfb, documented chrome inks). Site is light-only (0 dark-mode rules in style.css), so light theme is baked in. Palette validation: slots 1-2 of the reference palette are pre-validated in the skill's palette.md for the light surface (adjacent CVD dE 9.1, normal-vision 19.6); no new hues introduced, so no re-run needed. Chart (a) is single-series.
- Every number sourced from the claims-audit whitelist or the staged report JSONs:
  - 8 ratios in chart (a): report.json speed rows verbatim (0.604/0.876/0.927/0.953/0.994/0.999/1.013/1.025, all n=100).
  - dyn_add ranges: controller directive + ledger (1.133-1.144 -> 1.034-1.038).
  - Python ratios computed from report_py.json medians: 10.979/9.380=1.17, 17.226/9.380=1.84, 131.941/29.915=4.41, 235.534/29.915=7.87, 57.778/41.328=1.40, 53.584/41.328=1.30, 494.722/359.664=1.38, 303.483/359.664=0.84, 118.957/63.674=1.87, 220.139/63.674=3.46. n range 60-100 stated in footer (dim-32 cell with n=10 not used).
  - Hardware/WSL2/compiler from report.json meta (Ryzen 7 9800X3D, WSL2 true, g++ 13.3.0, sha 9cac562), disclosed in every chart footer and in the post's methodology paragraph.
  - Noise envelope 0.94-1.16 with 0.884-1.176 disclosure at both citation sites (chart subtitle + prose).
  - "5 of 7 conclusions corrected", 16-seed 1-ULP pynanoflann wobble, nanoflann 1.5.5 kernel attribution, GCC-AVX512-vs-LLVM-AVX2 rejection: all from the M2.6 record as summarized in docs/benchmarks.md and the audit.
- Honesty requirements: authorship disclosure is the opening paragraph; "what directing meant" paragraph included (specs, standards, contamination catch, traceability rule); game-contamination story told without uncited digits; the placeholder-numbers incident disclosed with its forensic clearance.
- Frontmatter mirrors 2026-06-01-cvpr-2026-survival-guide.md field-for-field (layout/title/description/date/author/permalink/updated/categories/tags/image/ogImage/twitterCard/excerpt/shortlink/readingTime/keywords/customSchema). categories: ["Research"] (an existing site category).

## Build verification

`npx eleventy`: "Copied 505 Wrote 85 files in 1.39 seconds", zero errors. Rendered HTML exists at _site/blog/posts/2026-08-30-flannrust-ai-agent-port.html with 3 flannrust img tags; all 3 SVGs copied to _site/assets/images/blog/.

Preview: build the site and open /blog/posts/2026-08-30-flannrust-ai-agent-port.html (or `npm run dev` in ~/dev/my_website on branch flannrust-post).

## Concerns

1. ogImage points at an SVG; some social scrapers (Twitter/X, Slack) won't render SVG og images. If share cards matter, rasterize flannrust-ratios.svg to a JPG/PNG cover before publishing and point image/ogImage at that.
2. The post links github.com/sitzikbs/flannrust, which is created-but-private (or still being created) at time of writing; the link 404s for the public until the repo flips public. Intentional per controller.
3. PyPI/crates.io wording says "not yet published; packaging verified" — matches T3's dry-run state; revisit if publishing lands before the post goes live.
4. readingTime "10 min read" is an estimate for ~2,100 words; adjust if the site has a convention.

## Fix round 1 (commit 47ae3a5 on flannrust-post)

All findings addressed:
1. CRITICAL 1.5.5 kernel: rewritten to the record — hypothesis tested and REFUTED (1.5.5 kernel 5-6% slower than 1.12.1 on that workload), miss not root-caused, open question narrowed to the Python binding layer, tracked as open item.
2. CRITICAL both losses: added batched knn workers=1 row to flannrust-python.svg (ck 1.42x = 399.748/282.147, py 0.97x = 274.114/282.147 from report_py.json); prose now "two honest losses" with recorded ranges (batched 6-22% across recorded runs / ~3% fresh; dim8 up to 25% recorded / ~16% fresh).
3. dyn_add: both fixes credited (Vec capacity -> ~1.06, then 4-wide min/max unroll -> 1.034-1.038); prose says 11-15% before.
4. Incident attribution: "the agent wrote fabricated placeholder numbers... It then reported this itself, unprompted"; own paragraph in rewritten trust section.
5. flannrust-dynadd.svg: before range now canonical 1.112-1.149, bar to midpoint 1.1305, whiskers at endpoints, era label "measured in M2.6, before vs after the fix commits", no M2.5 tag.
6. Verification matrix corrected: dims 2-32, leaf 1-64, 60 seeded queries per configuration; dim-64/leaf-1024 explicitly labeled benchmark sweeps; "200 seeded datasets" quote replaced with the 60-query phrasing.
7. Parallel-build row labeled in chart ("flannrust parallel vs their single-threaded builds") + a dedicated honesty paragraph in prose (neither library parallelizes construction; same-rules row is the single-threaded one).
8. 19% -> "about 16% faster in the fresh run" (303.483/359.664 = 0.844).
9. Determinism claim fixed: both libraries partition-before-spawn; differentiator stated as lock-free local arenas / no contention.
10. flannrust-ratios.png rasterized via headless Chrome (1424x851, cropped body margin); frontmatter image/ogImage now point at the PNG.
11. 7.87x label moved outside its bar (axis re-scaled to 0-8.5 domain).
12. Ticks now 0/1/2/4/6/8 at even 2x intervals past 2 (1x kept as the parity marker).
13. L2 kernel: "four components at a time, then the leftover dimensions in a fixed descending order".
14. Nodes: "20 bytes for f32 data (32 for f64)".
15. cKDTree wording no longer reads as a range; per-row margins listed.
16. height attrs on all three img tags (430/264/422).
18. Chart footer: Python 3.12.11, SciPy 1.18.1, pynanoflann 0.10.0 added.
Quality: antithesis constructions cut to ~4 across the post (kept "tests are the authority, not the author", the opening "Not approximately...", "evidence instead of confidence"); "Three design choices" -> "A few design choices"; trust section rewritten shorter with the incident in its own paragraph.
RUSTFLAGS check: report_py.json meta.rustflags is empty for the bench process; wheel build-time flags unprovable from the JSON, so the native-tuning claim is scoped to the Rust chart ("Everything in this section...") and the Python section states outright: stock pip installs vs locally built wheel, nothing tuned for the CPU.
SVG footers were clipping at 720 width; wrapped to extra lines, viewBoxes now 430/264/422 tall. Site rebuilt: "Wrote 85 files", zero errors; PNG + 3 SVGs in _site.
