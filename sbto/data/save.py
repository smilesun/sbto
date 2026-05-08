import os
import numpy as np
import shutil
import numpy.typing as npt
from dataclasses import asdict
from typing import List
import mujoco

from sbto.tasks.task_mj import TaskMj
from sbto.sim.sim_mj_rollout import SimMjRollout
from sbto.tasks.task_base import OCPBase
from sbto.solvers.solver_base import SolverState
from sbto.utils.plotting import plot_contact_plan, plot_costs, plot_mean_cov, plot_state_control, plot_knot_scatter
from sbto.utils.viewer import render_and_save_trajectory
from sbto.data.utils import solver_state_path_from_rundir, create_dirs
from sbto.data.postprocess import split_x_traj
from sbto.data.aggregate import get_top_samples, get_top_samples_per_population
from sbto.data.constants import *
from sbto.evaluation.trajectory_diversity import (
    save_every_n_iteration_diversity_analysis,
    save_last_iteration_diversity_analysis,
)
from sbto.evaluation.fall_detection import check_trajectory_file, detect_fall
from sbto.evaluation.trajectory_cluster_diversity import (
    save_cluster_diversity_analysis,
    save_every_n_iteration_cluster_diversity_analysis,
)

Array = npt.NDArray[np.float64]

def save_all_samples_and_cost(
    dir_path: str,
    samples,
    costs,
    ) -> None:
    np.savez(
        os.path.join(dir_path, f"{ALL_SAMPLES_COSTS_FILENAME}.npz"),
        samples=samples,
        costs=costs,
    )

def save_solver_state(
    dir_path: str,
    state: SolverState,
    suffix: str = ""
    ) -> None:
    solver_state_path = solver_state_path_from_rundir(dir_path, suffix)
    np.savez(solver_state_path, **asdict(state))

def save_plots(
    dir_path: str,
    task: TaskMj,
    time,
    x_traj,
    u_traj,
    obs_traj,
    knots,
    mean_knots,
    cov_knots,
    all_costs,
    ) -> None:

    Nu, Nq = task.mj_scene.Nu, task.mj_scene.Nq

    plot_mean_cov(
        time,
        mean_knots,
        knots,
        cov_knots,
        u_traj,
        Nu=Nu,
        save_dir=dir_path,
    )

    plot_costs(
        all_costs,
        save_dir=dir_path
        )

    plot_state_control(
        time,
        x_traj,
        u_traj,
        knots,
        Nq,
        Nu,
        save_dir=dir_path
        )
    
    contact_realized = task.get_contact_status(obs_traj)

    if len(contact_realized) > 0:
        contact_realized[contact_realized > 1] = 1
        contact_plan = task.contact_plan if hasattr(task, "contact_plan") else None
        plot_contact_plan(
            contact_realized,
            contact_plan,
            dt=task.mj_scene.dt,
            save_dir=dir_path
        )

def copy_hydra_config(hydra_rundir: str, dst_path: str):
    hydra_cfg_dir = f"{hydra_rundir}/{HYDRA_CFG}" if hydra_rundir else ""
    if os.path.exists(hydra_cfg_dir):
        shutil.copytree(hydra_cfg_dir, f"{dst_path}/{HYDRA_CFG}")

def save_mj_model(dir_path: str, mj_spec: mujoco.MjSpec):
    file_name = os.path.join(dir_path, f"{MJ_MODEL_NAME}.xml")
    with open(file_name, "w") as f:
        f.write(mj_spec.to_xml())

def save_results(
    data_dir: str,
    sim: SimMjRollout,
    task: OCPBase,
    solver_state_0: SolverState,
    solver_state_final: SolverState,
    all_samples: Array,
    best_samples_it: List[Array],
    all_costs: Array,
    exp_name: str = "",
    description: str = "",
    hydra_rundir: str = "",
    save_fig: bool = True,
    save_video: bool = True,
    save_samples_costs: bool = True,
    save_best_samples_it: bool = True,
    multiple_shooting: bool = False,
    split_state: bool = False,
    save_top: float = 0.,
    n_last_it: int = 0,
    remove_keys: List[str] = [],
    diversity_plot_every: int = 0,
    knot_cost_threshold: float = float("inf"),
    fall_floor_height: float = 0.0,
    fall_drop_threshold: float = 0.25,
    result_dir: str = "",
    num_populations: int = 0,
    ) -> str:
    if not result_dir:
        exp_name = task.__class__.__name__ if not exp_name else exp_name
        result_dir = create_dirs(exp_name, data_dir, description)

    # Save config
    copy_hydra_config(hydra_rundir, result_dir)

    # Save mj model
    save_mj_model(result_dir, sim.mj_scene.edit.mj_spec)

    # Save inital and final solver state
    if solver_state_0:
        save_solver_state(result_dir, solver_state_0, INITIAL_SOLVER_STATE_SUFFIX)
    save_solver_state(result_dir, solver_state_final, FINAL_SOLVER_STATE_SUFFIX)
    
    if n_last_it > 0:
        N_it_samples = n_last_it
        all_samples = all_samples[-N_it_samples:]
    else:
        N_it_samples = all_samples.shape[0]
    last_costs = all_costs[-N_it_samples:]

    # Save all samples and costs from the optimization
    if save_samples_costs:
        if n_last_it > 0:
            print(f"Saving all samples and costs from last {n_last_it} iteration.")
        else:
            print("Saving all samples and costs.")
        save_all_samples_and_cost(result_dir, all_samples, last_costs)

    # Save best samples per iterations
    if save_best_samples_it:
        data = {str(i) : sample for i, sample in enumerate(best_samples_it)}
        data[KEY_COST] = np.min(all_costs, axis=0)
        np.savez_compressed(
            os.path.join(result_dir, f"{BEST_SAMPLES_IT_FILENAME}.npz"),
            **data
        )

    # Rollout best trajectories (with initial states)
    N_top_samples = 1 # Save the best one by default
    if save_top > 0.:
        # How many top traj to save
        if save_top >= 1:
            N_samples = last_costs.size
            N_top_samples = min(int(save_top), N_samples)
        else:
            percentile = save_top
            threshold = np.percentile(last_costs, percentile)
            top_mask = last_costs <= threshold
            N_top_samples = np.sum(top_mask)
 
    top_samples, top_costs = get_top_samples(last_costs, all_samples, N_top_samples)

    if multiple_shooting:
        x_shooting = task.ref.x[sim.t_knots]
        t, x_traj, qdes_traj, obs_traj = map(np.squeeze, sim.rollout_multiple_shooting(top_samples, x_shooting, with_x0=True))
    else:
        t, x_traj, qdes_traj, obs_traj = map(np.squeeze, sim.rollout(top_samples, with_x0=True))
    
    # Sanity check
    # cost = task.cost(x_traj[None, 1:, :], top_samples, obs_traj[None, :, :])
    # print(f"[{description or 'Unnamed'}] Best cost: {np.min(cost)}")

    print(f"[{description or 'Unnamed'}] Best cost: {solver_state_final.min_cost_all}")

    # By default all data keys are saved
    data_traj = {
        KEY_TIME: t,
        KEY_FULL_STATE: x_traj,
        KEY_PD_TARGET: qdes_traj,
        KEY_OBS: obs_traj,
        KEY_COST: top_costs,
        KEY_STEP_KNOTS: sim.t_knots,
    }

    # Split state
    if split_state:
        splitted_data = split_x_traj(x_traj, sim.mj_scene)
        data_traj.update({
            k: v
            for k, v in splitted_data.items()
            if not k in data_traj
        })

    # Save best
    best_trajectory_path = os.path.join(result_dir, f"{BEST_TRAJECTORY_FILENAME}.npz")
    file_path = best_trajectory_path
    if N_top_samples == 1:
        best_data = data_traj.copy()
    else:
        arg_min_cost = np.argmin(top_costs)
        best_data = {k: np.squeeze(v[arg_min_cost]) for k, v in data_traj.items()}
    np.savez_compressed(
        file_path,
        **{
            k: v for k, v in best_data.items()
            if k not in remove_keys
        }
    )
    
    # Remove keys from data
    for k in remove_keys:
        if k in data_traj.keys():
            del data_traj[k]

    # Save top trajectories
    if N_top_samples > 1:
        print(f"Saving top {N_top_samples} trajectories.")
        file_path = os.path.join(result_dir, f"{TOP_TRAJECTORIES_FILENAME}.npz")
        np.savez_compressed(
            file_path,
            **data_traj
        )

    # Save per-subpopulation top-1 trajectories (SV-CMA-ES)
    if num_populations > 1 and all_samples.shape[0] > 0:
        pop_samples, pop_costs, pop_labels = get_top_samples_per_population(
            last_costs, all_samples, num_populations, k_per_pop=1
        )
        if multiple_shooting:
            x_shooting = task.ref.x[sim.t_knots]
            _, x_pop, _, _ = map(np.squeeze, sim.rollout_multiple_shooting(pop_samples, x_shooting, with_x0=True))
        else:
            _, x_pop, _, _ = map(np.squeeze, sim.rollout(pop_samples, with_x0=True))
        if x_pop.ndim == 2:
            x_pop = x_pop[None]
        pop_data = {
            KEY_FULL_STATE: x_pop,
            KEY_COST: pop_costs,
            "population_index": pop_labels,
        }
        for k in remove_keys:
            pop_data.pop(k, None)
        file_path = os.path.join(result_dir, "top_trajectories_per_population.npz")
        np.savez_compressed(file_path, **pop_data)
        print(f"Saved per-population top-1 trajectories ({num_populations} populations) -> {file_path}")

    print(
        "Diversity save debug:",
        f"save_fig={save_fig},",
        f"all_samples_shape={all_samples.shape},",
        f"all_costs_shape={all_costs.shape},",
        f"diversity_plot_every={diversity_plot_every},",
        f"result_dir={result_dir}"
    )

    # Save all figures
    if save_fig:
        best_knots = solver_state_final.best_all
        save_plots(
            result_dir,
            task,
            best_data[KEY_TIME],
            best_data[KEY_FULL_STATE],
            best_data[KEY_PD_TARGET],
            best_data[KEY_OBS],
            best_knots,
            solver_state_final.mean,
            solver_state_final.cov,
            all_costs,
        )
        if all_samples.shape[0] > 0 and all_costs.shape[0] > 0:
            fall_mask = None
            if fall_floor_height > 0.0:
                _, x_traj_all, _, _ = sim.rollout(all_samples[-1], with_x0=True)
                root_pos_all = split_x_traj(x_traj_all, sim.mj_scene).get(KEY_ROOT_POS)
                if root_pos_all is not None:
                    dummy_time = np.arange(root_pos_all.shape[1], dtype=float)
                    fall_mask = np.array([
                        detect_fall(root_pos_all[i], dummy_time,
                                    fall_floor_height, fall_drop_threshold).fallen
                        for i in range(root_pos_all.shape[0])
                    ])
            plot_knot_scatter(
                all_samples[-1],
                all_costs[-1],
                sim.t_knots,
                sim.mj_scene.Nu,
                save_dir=result_dir,
                cost_threshold=knot_cost_threshold,
                fall_mask=fall_mask,
            )
            save_last_iteration_diversity_analysis(
                result_dir,
                sim,
                all_samples[-1],
                all_costs[-1],
            )
            save_cluster_diversity_analysis(
                result_dir,
                all_samples[-1],
                all_costs[-1],
                sim,
            )
            if diversity_plot_every > 0:
                save_every_n_iteration_diversity_analysis(
                    result_dir,
                    sim,
                    all_samples,
                    all_costs,
                    diversity_plot_every,
                )
                save_every_n_iteration_cluster_diversity_analysis(
                    result_dir,
                    sim,
                    all_samples,
                    all_costs,
                    diversity_plot_every,
                )

    # Fall detection (always run on the single best trajectory)
    fall_report = check_trajectory_file(best_trajectory_path)
    print(fall_report)

    # Save video rendering
    if save_video:
        # Best trajectory
        render_and_save_trajectory(
            sim.mj_scene.mj_model,
            sim.mj_scene.mj_data,
            best_data[KEY_TIME],
            best_data[KEY_FULL_STATE],
            save_path=result_dir,
        )
        # Top-k trajectories (when more than one was requested)
        if N_top_samples > 1:
            x_top = x_traj if x_traj.ndim == 3 else x_traj[None]  # (K, T, nx)
            t_single = best_data[KEY_TIME]  # (T,) — all trajectories share the same time axis
            for rank in range(x_top.shape[0]):
                render_and_save_trajectory(
                    sim.mj_scene.mj_model,
                    sim.mj_scene.mj_data,
                    t_single,
                    x_top[rank],
                    save_path=os.path.join(result_dir, f"trajectory_vis_top_{rank}.mp4"),
                )

    return result_dir
