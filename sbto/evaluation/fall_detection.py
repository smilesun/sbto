"""
Programmatic fall detection for saved humanoid trajectories.

A "fall" is defined by two independent criteria:
  1. Root height drops below an absolute floor (robot hits the ground).
  2. Root height drops sharply relative to its initial value (robot is
     actively falling even if it hasn't fully collapsed yet).

Both thresholds are conservative: they will not false-positive on normal
crouching or task-driven height variation, but will catch any trajectory
where the robot loses balance.
"""

from __future__ import annotations
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

Array = npt.NDArray[np.float64]

# Height of the root (pelvis) when the G1 stands upright.  Trajectories with
# min height below this value have either fallen or are in an extreme crouch.
DEFAULT_FLOOR_HEIGHT: float = 0.40   # metres

# A drop larger than this relative to the first frame indicates active falling.
DEFAULT_DROP_THRESHOLD: float = 0.25  # metres


@dataclass
class FallReport:
    fallen: bool
    fall_time: float | None      # first time step where fall is detected
    min_height: float
    initial_height: float
    max_drop: float              # largest drop from initial height
    floor_violated: bool         # min_height < floor_height threshold
    drop_violated: bool          # max_drop > drop_threshold
    floor_height: float
    drop_threshold: float

    def __str__(self) -> str:
        status = "FALLEN" if self.fallen else "UPRIGHT"
        lines = [
            f"[{status}]",
            f"  min_height    = {self.min_height:.3f} m  (floor={self.floor_height:.3f} m)",
            f"  initial_height= {self.initial_height:.3f} m",
            f"  max_drop      = {self.max_drop:.3f} m  (threshold={self.drop_threshold:.3f} m)",
        ]
        if self.fallen:
            lines.append(f"  fall detected at t = {self.fall_time:.3f} s")
        return "\n".join(lines)


def detect_fall(
    root_pos: Array,          # (T, 3)  root position [x, y, z]
    time: Array,              # (T,)
    floor_height: float = DEFAULT_FLOOR_HEIGHT,
    drop_threshold: float = DEFAULT_DROP_THRESHOLD,
) -> FallReport:
    """
    Detect whether the humanoid falls during the trajectory.

    Parameters
    ----------
    root_pos : (T, 3) array — root (pelvis) positions over time.
    time     : (T,) array  — corresponding timestamps.
    floor_height   : root z below this → floor criterion triggered.
    drop_threshold : root z drops more than this from t=0 → drop criterion triggered.

    Returns
    -------
    FallReport with scalar diagnostics and a boolean verdict.
    """
    z = root_pos[:, 2]
    initial_height = float(z[0])
    min_height = float(z.min())
    max_drop = float((initial_height - z).max())

    floor_violated = min_height < floor_height
    drop_violated = max_drop > drop_threshold
    fallen = floor_violated or drop_violated

    fall_time = None
    if fallen:
        # First timestep where either criterion is met.
        floor_mask = z < floor_height
        drop_mask = (initial_height - z) > drop_threshold
        combined = floor_mask | drop_mask
        first_idx = int(np.argmax(combined))
        fall_time = float(time[first_idx])

    return FallReport(
        fallen=fallen,
        fall_time=fall_time,
        min_height=min_height,
        initial_height=initial_height,
        max_drop=max_drop,
        floor_violated=floor_violated,
        drop_violated=drop_violated,
        floor_height=floor_height,
        drop_threshold=drop_threshold,
    )


def check_trajectory_file(
    npz_path: str,
    floor_height: float = DEFAULT_FLOOR_HEIGHT,
    drop_threshold: float = DEFAULT_DROP_THRESHOLD,
) -> FallReport:
    """Convenience wrapper that loads a saved best_trajectory.npz and checks it."""
    d = dict(np.load(npz_path))
    root_pos = d["root_pos"]    # (T, 3)
    time = d["time"]            # (T,)
    return detect_fall(root_pos, time, floor_height, drop_threshold)
