import hydra
from hydra.utils import instantiate
from typing import Optional
import copy
import os
import numpy as np
from functools import partial

from sbto.sim.sim_base import SimRolloutBase
from sbto.tasks.task_base import OCPBase
from sbto.tasks.task_mj_ref import TaskMjRef
from sbto.solvers.solver_base import SamplingBasedSolver, SolverState
from sbto.run.optimize import optimize_single_shooting, optimize_mutiple_shooting, optimize_incremental_opt
from sbto.run.save import save_results, get_final_state_from_rundir

def optimize_and_save_data(
    cfg,
    sim: SimRolloutBase,
    task: OCPBase,
    solver: SamplingBasedSolver,
    hydra_rundir: str = "",
    solver_state_0: Optional[SolverState] = None,
    ) -> None:

    # Copy initial state
    if solver_state_0:
        solver_state_0 = copy.deepcopy(solver_state_0)
    else:
        solver_state_0 = copy.deepcopy(solver.state)

    # Multiple_shooting
    if cfg.warm_start.multiple_shooting:
        if not isinstance(task, TaskMjRef):
            raise ValueError("Task should be an instance of TaskMjRef (with reference)")
        optimizer_fn = optimize_mutiple_shooting

    # Incremental opt
    elif cfg.warm_start.incremental:
        optimizer_fn = partial(
            optimize_incremental_opt,
            N_max_it_per_knots=cfg.warm_start.N_max_incr,
            min_std_next=cfg.warm_start.min_std_next,
        )

    # Single shooting
    else:
        optimizer_fn = optimize_single_shooting
    
    solver_state_final, all_samples, all_costs = optimizer_fn(
        sim,
        task,
        solver,
        solver_state_0
    )

    save_results(
        sim,
        task,
        solver_state_0,
        solver_state_final,
        all_samples,
        all_costs,
        cfg.description,
        hydra_rundir,
        cfg.save_fig,
        cfg.warm_start.multiple_shooting,
    )

def instantiate_from_cfg(cfg):
    sim = instantiate(cfg.task.sim)
    task = instantiate(cfg.task, sim=sim)
    solver = instantiate(cfg.solver, D=sim.Nvars_u)
    return sim, task, solver

def get_initial_state_solver_from_ref(sim, task, solver):
    if not isinstance(task, TaskMjRef):
        print("Task has no reference.")
        return None
    qpos_from_ref = task.ref.act_qpos[sim.t_knots, :]
    pd_knots_from_ref = sim.scaling.inverse(qpos_from_ref).reshape(-1)
    solver_state_0 = solver.init_state(mean=pd_knots_from_ref)
    return solver_state_0

def get_warm_start_state_solver(cfg, sim, task, solver) -> SolverState:
    # Set initial solver state
    solver_state_0 = None
    if cfg.init_knots_from_ref and isinstance(task, TaskMjRef):
        solver_state_0 = get_initial_state_solver_from_ref(sim, task, solver)

    if cfg.warm_start.rundir and os.path.exists(cfg.warm_start.rundir):
        solver_state_0 = get_final_state_from_rundir(cfg.warm_start.rundir, solver)
        print(solver_state_0.mean.shape)
        solver.reset_min_cost_best(solver_state_0)

        if cfg.warm_start.add_cov_diag > 0.:
            solver.init_state()
            N = solver_state_0.mean.shape[0]
            solver_state_0.cov += cfg.warm_start.add_cov_diag * np.eye(N)

    return solver_state_0

def set_cfg_warm_start(cfg):
    cfg_ws = copy.deepcopy(cfg)
    WARM_START_MULTIPLE_SHOOTING = "ws_ms"
    WARM_START_INCREMENTAL = "ws_incr"

    # Update description
    sep = "_" if cfg.description else ""
    if cfg_ws.warm_start.incremental:
        cfg_ws.description += sep + WARM_START_INCREMENTAL
    
    elif cfg_ws.warm_start.multiple_shooting:
        cfg_ws.description += sep + WARM_START_MULTIPLE_SHOOTING
    return cfg_ws

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    hydra_rundir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    sim, task, solver = instantiate_from_cfg(cfg)

    # Warm start
    if (
        cfg.warm_start.multiple_shooting or
        cfg.warm_start.incremental
        ):
        
        cfg_ws = set_cfg_warm_start(cfg)

        # Update solver Nit
        _N_it = cfg_ws.solver.cfg.N_it
        if cfg_ws.warm_start.N_it > 0:
            solver.cfg.N_it = cfg_ws.warm_start.N_it

        solver_state_0 = get_warm_start_state_solver(cfg, sim, task, solver)
        rundir = optimize_and_save_data(
            cfg_ws,
            sim,
            task,
            solver,
            hydra_rundir,
            solver_state_0=solver_state_0,
        )
        cfg.warm_start.rundir = rundir

        # Set back the number of solver iterations
        solver.cfg.N_it = _N_it
        # Reset warm start params
        cfg.warm_start.multiple_shooting = False
        cfg.warm_start.incremental = False

    # Optimize single shooting
    if solver.cfg.N_it > 0:
        solver_state_0 = get_warm_start_state_solver(cfg, sim, task, solver)
        rundir = optimize_and_save_data(
            cfg,
            sim,
            task,
            solver,
            hydra_rundir,
            solver_state_0,
        )
        
if __name__ == "__main__":
    main()