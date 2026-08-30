#!/usr/bin/env python3
"""Portable bench runner: captures host specs + runs the Rust (and optionally
Python) benchmark suites, emitting one self-describing JSON bundle per host."""
import argparse, datetime, json, os, platform, shutil, subprocess, sys

def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)

def cpu_model():
    s = platform.system()
    if s == "Linux":
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    if s == "Darwin":
        r = sh(["sysctl", "-n", "machdep.cpu.brand_string"])
        if r.returncode == 0:
            return r.stdout.strip()
    if s == "Windows":
        r = sh(["wmic", "cpu", "get", "name"])
        if r.returncode == 0:
            lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
            if len(lines) > 1:
                return lines[1]
    return platform.processor() or "unknown"

def tool_version(names):
    for n in names:
        if shutil.which(n):
            r = sh([n, "--version"])
            if r.returncode == 0:
                return r.stdout.splitlines()[0]
    return None

def host_specs():
    la = os.getloadavg() if hasattr(os, "getloadavg") else None
    return {
        "platform": platform.platform(),
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "wsl": "microsoft" in platform.release().lower(),
        "python": sys.version.split()[0],
        "rustc": tool_version(["rustc"]),
        "cxx": tool_version(["c++", "g++", "clang++", "cl"]),
        "loadavg": la,
        "git_sha": sh(["git", "rev-parse", "--short", "HEAD"]).stdout.strip(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", help="venv python for bench_py (optional)")
    ap.add_argument("--out")
    ap.add_argument("--force", action="store_true", help="skip idle-host check")
    a = ap.parse_args()
    specs = host_specs()
    if not a.force and specs["loadavg"] and specs["loadavg"][0] > 0.5:
        sys.exit(f"host not idle (loadavg {specs['loadavg'][0]}); close apps or --force")
    env = dict(os.environ, RUSTFLAGS="-C target-cpu=native")
    r = sh(["cargo", "run", "-q", "-p", "xval", "--release", "--example", "report_data"], env=env)
    if r.returncode != 0:
        sys.exit(f"report_data failed:\n{r.stderr[-2000:]}")
    bundle = {"host": specs, "rust_report": json.loads(r.stdout), "python_report": None}
    if a.python:
        rp = sh([a.python, "crates/flannrust-py/python/bench/bench_py.py"])
        if rp.returncode != 0:
            sys.exit(f"bench_py failed:\n{rp.stderr[-2000:]}")
        bundle["python_report"] = json.loads(rp.stdout)
    out = a.out or f"bench-{platform.node()}-{datetime.date.today():%Y%m%d}.json"
    with open(out, "w") as f:
        json.dump(bundle, f, indent=1)
    print(out)

if __name__ == "__main__":
    main()
