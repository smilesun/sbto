"""
Render trajectory videos for completed hpsearch trials that have no video yet.

Usage (from repo root):
    uv run python benchmark/render_hpsearch_videos.py
"""

import glob
import os
import numpy as np
import mujoco
from sbto.utils.viewer import render_and_save_trajectory

TRIAL_GLOB = "outputs/G1RobotObjRef/*hpsearch*"


def reconstruct_state(d: dict) -> np.ndarray:
    """Reassemble qpos+qvel from the split keys saved by split_x_traj."""
    qpos = np.concatenate([
        d["root_pos"],
        d["root_rot"],
        d["dof_pos"],
        d["object_pos"],
        d["object_rot"],
    ], axis=-1)
    qvel = np.concatenate([
        d["root_lin_vel"],
        d["root_ang_vel"],
        d["dof_vel"],
        d["object_lin_vel"],
        d["object_ang_vel"],
    ], axis=-1)
    return np.concatenate([qpos, qvel], axis=-1)


def render_trial(result_dir: str) -> None:
    video_path = os.path.join(result_dir, "trajectory_vis.mp4")
    if os.path.exists(video_path):
        print(f"  already exists, skipping: {video_path}")
        return

    traj_path = os.path.join(result_dir, "best_trajectory.npz")
    model_path = os.path.join(result_dir, "mj_model.xml")

    if not os.path.exists(traj_path):
        print(f"  missing best_trajectory.npz, skipping: {result_dir}")
        return
    if not os.path.exists(model_path):
        print(f"  missing mj_model.xml, skipping: {result_dir}")
        return

    d = dict(np.load(traj_path))
    t = d["time"]
    x_traj = reconstruct_state(d)

    assets_dir = os.path.join(os.path.dirname(__file__), "..", "sbto", "models", "unitree_g1", "assets")
    assets_dir = os.path.realpath(assets_dir)
    with open(model_path) as f:
        xml = f.read()
    xml = xml.replace('meshdir="assets/"', f'meshdir="{assets_dir}/"')
    mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(mj_model)

    print(f"  rendering {os.path.basename(result_dir)} ...")
    render_and_save_trajectory(mj_model, mj_data, t, x_traj, save_path=result_dir)


def main():
    dirs = sorted(glob.glob(TRIAL_GLOB))
    if not dirs:
        print("No hpsearch result directories found.")
        return

    for d in dirs:
        print(d)
        render_trial(d)


if __name__ == "__main__":
    main()
