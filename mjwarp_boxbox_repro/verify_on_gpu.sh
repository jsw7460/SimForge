#!/usr/bin/env bash
# Verify the box_box edge-axis fix on this machine, end to end.
#
# Three things have to hold for the PR to be worth submitting:
#   1. the new test FAILS on upstream main   (the bug is real on this platform)
#   2. the new test PASSES with the fix      (the fix works on this platform)
#   3. the rest of the suite is unaffected
#
# Point 1 is the one that cannot be checked from a Mac: the bug turns on which
# way a float32 tie rounds, and that can differ between architectures.
#
#   bash verify_on_gpu.sh
#
# Nothing is silenced. If the environment cannot be built this stops there and
# says so, rather than reporting a test result that means nothing.
set -u

DIR=~/workspace/mjwarp-pr
BRANCH=box-box-edge-axis-float32-tie
SRC=mujoco_warp/_src/collision_primitive_core.py
TEST=mujoco_warp/_src/collision_primitive_core_test.py

die() { echo; echo "STOPPED: $*"; exit 1; }

# ---- the clone ------------------------------------------------------------
if [ -d "$DIR/.git" ]; then
  cd "$DIR" || die "cannot enter $DIR"
  git fetch -q origin "$BRANCH" || die "git fetch failed"
  git checkout -q -B "$BRANCH" FETCH_HEAD || die "git checkout failed"
else
  git clone -b "$BRANCH" https://github.com/jsw7460/mujoco_warp.git "$DIR" || die "git clone failed"
  cd "$DIR" || die "cannot enter $DIR"
fi
echo "repo   $(pwd)"
echo "commit $(git rev-parse --short HEAD)  $(git log -1 --format=%s)"

# ---- the environment ------------------------------------------------------
# uv sync only checks that the right versions are recorded, not that they still
# import; a half-written package passes the audit and fails at run time.
show_env() {
  uv run python -c "
import sys
print('python     ', sys.version.split()[0], sys.executable)
import numpy;       print('numpy      ', numpy.__version__, numpy.__file__)
import mujoco;      print('mujoco     ', mujoco.__version__)
import warp as wp;  wp.init(); print('warp       ', wp.__version__, '->', wp.get_device())
import mujoco_warp; print('mujoco_warp', mujoco_warp.__file__)
"
}

echo
echo "---- uv sync ----"
uv sync --extra dev || die "uv sync failed (see above)"

echo
echo "---- what the environment actually has ----"
if ! show_env; then
  echo
  echo "---- that environment is broken; here is what it looks like ----"
  echo "PYTHONPATH = ${PYTHONPATH-<unset>}"
  echo "PYTHONHOME = ${PYTHONHOME-<unset>}"
  echo "VIRTUAL_ENV = ${VIRTUAL_ENV-<unset>}"
  echo "UV_CACHE_DIR = ${UV_CACHE_DIR-<unset>}"
  SP=$(echo .venv/lib/python3*/site-packages)
  echo "site-packages: $SP"
  ls -d "$SP"/numpy* 2>/dev/null
  find "$SP/numpy" -maxdepth 2 -name "version.py" -o -maxdepth 1 -name "__init__.py" 2>/dev/null | head
  echo "numpy dist-info version: $(cat "$SP"/numpy-*.dist-info/METADATA 2>/dev/null | grep -m1 ^Version:)"
  echo "compiled cores present:"
  ls "$SP"/numpy/_core/_multiarray_umath*.so 2>/dev/null || echo "  (none -- the compiled half never got installed)"

  echo
  echo "Rebuilding from a clean cache, once."
  rm -rf .venv || die "could not remove .venv"
  uv cache clean numpy || echo "  (uv cache clean numpy returned non-zero; continuing)"
  uv sync --extra dev --reinstall-package numpy || die "uv sync failed on the rebuilt venv (see above)"
  echo
  show_env || die "numpy still will not import after a clean cache and a fresh venv.

That is not this PR -- it is the machine. Two things usually do it:
  - a PYTHONPATH entry ahead of the venv, shown above;
  - a uv cache on a filesystem that damaged the wheel, in which case
    'uv cache clean && rm -rf .venv && uv sync --extra dev' from this directory
    is the wider version of what was just tried."
fi

# ---- 1: the bug ------------------------------------------------------------
echo
echo "======== [1/3] upstream main, WITHOUT the fix -- the test must FAIL ========"
git checkout -q HEAD~1 -- "$SRC" || die "could not revert $SRC"
uv run pytest "$TEST" -k level_boxes -q
BEFORE=$?

# ---- 2: the fix ------------------------------------------------------------
echo
echo "======== [2/3] WITH the fix -- the test must PASS ========"
git checkout -q HEAD -- "$SRC" || die "could not restore $SRC"
uv run pytest "$TEST" -k level_boxes -q
AFTER=$?

# ---- 3: everything else ----------------------------------------------------
echo
echo "======== [3/3] the rest of the suite ========"
uv run pytest -n 8 -q 2>&1 | grep -E "^FAILED|passed|failed|error"

# ---- verdict ---------------------------------------------------------------
echo
echo "=================================================================="
if [ "$BEFORE" -ne 0 ] && [ "$AFTER" -eq 0 ]; then
  echo "  Reproduced here, and fixed here. The PR holds on this platform."
elif [ "$BEFORE" -eq 0 ] && [ "$AFTER" -eq 0 ]; then
  echo "  The test passed WITHOUT the fix on this machine."
  echo "  The bug is real -- 92 of 2000 poses on the Mac -- but this particular"
  echo "  pose does not round the wrong way here, so the test would not guard"
  echo "  the regression on x86 CI. The pose needs to be found on this machine"
  echo "  instead; mjwarp_boxbox_repro/main.py is the search."
else
  echo "  The test FAILED with the fix applied. Do not submit; report the output."
fi
echo "=================================================================="
