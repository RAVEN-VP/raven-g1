
"""RAVEN reverse static sweep and return-to-centre diagnostic.

Requires these existing files in the same scripts directory:
    raven_g1_actor_tracking.py
    raven_g1_static_framing_sweep.py

Keeps the calibrated controller and RGB camera unchanged.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

SWEEP_SCRIPT = (
    SCRIPTS_DIR / "raven_g1_static_framing_sweep.py"
)

TRACKER_SCRIPT = (
    SCRIPTS_DIR / "raven_g1_actor_tracking.py"
)

for required_file in (SWEEP_SCRIPT, TRACKER_SCRIPT):
    if not required_file.is_file():
        raise SystemExit(
            f"Missing required script: {required_file}"
        )

# Importing the existing sweep initialises Isaac Sim,
# but does not start its simulation loop.

spec = importlib.util.spec_from_file_location(
    "raven_static_sweep_original",
    SWEEP_SCRIPT,
)

if spec is None or spec.loader is None:
    raise SystemExit(f"Cannot load: {SWEEP_SCRIPT}")

sweep = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sweep
spec.loader.exec_module(sweep)

# Reverse the successful sweep, then return to centre.

sweep.POSITIONS_M = (
    +0.45,
    +0.30,
    +0.15,
     0.00,
    -0.15,
    -0.30,
    -0.45,
     0.00,
)

# Separate output directory: preserve previous results.

sweep.OUT = (
    Path.cwd()
    / "raven_tracking_results"
    / "reverse_static_framing_sweep"
    / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)

sweep.OUT.mkdir(
    parents=True,
    exist_ok=False,
)

sweep.base.args.output_dir = str(sweep.OUT)

print("\n" + "=" * 72)
print(" RAVEN: REVERSE STATIC SWEEP")
print("=" * 72)

print(
    "Sequence (m): "
    + ", ".join(
        f"{value:+.2f}"
        for value in sweep.POSITIONS_M
    )
)

print(
    f"Hold duration: {sweep.HOLD_S:.1f}s"
)

print(
    f"Excluded settling: "
    f"{sweep.DISCARD_INITIAL_S:.1f}s"
)

print(f"Output: {sweep.OUT}")

print(
    "Original controller, grasp and optical "
    "calibration remain unchanged.\n"
)

try:
    sweep.base.main()
finally:
    try:
        sweep.summarise()
    finally:
        sweep.base.simulation_app.close()
