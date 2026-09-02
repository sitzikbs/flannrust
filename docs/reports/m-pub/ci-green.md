# M-pub CI evidence

First green CI run on branch m-pub (commit b763c27), 2026-09-02T06:08Z:

- CI @ b763c27 (m-pub): success — https://github.com/sitzikbs/flannrust/actions/runs/33597064707
- CI @ 43ba71e (m-pub): failure — https://github.com/sitzikbs/flannrust/actions/runs/33596598800

Jobs of the green run: test success, miri success, python success.

Run 1 (43ba71e) failed in the python job — maturin develop requires a virtualenv
on hosted runners; fixed in b763c27 by creating a job-level venv exported via
GITHUB_ENV/GITHUB_PATH. wheels.yml (workflow_dispatch) can only run once the
workflow exists on the default branch, so the three-OS wheel matrix is validated
post-merge.

## Post-merge (main, 2026-09-02)

- CI on main after merge 86b0eb5 and trigger-trim 24685f8: success (both runs).
- wheels.yml dispatch run 33599537545: success — artifacts wheels-ubuntu-latest,
  wheels-macos-14, wheels-windows-latest (abi3-py39). The Linux CI wheel
  (manylinux2014_x86_64) was downloaded and smoke-tested in a fresh venv:
  `KDTree(...).query(...)` printed `smoke ok (5, 3) (5, 3)`.
