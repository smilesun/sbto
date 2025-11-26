import time
import copy
import numpy as np
import numpy.typing as npt
from tqdm import trange
from typing import Tuple, Optional, Any

from sbto.sim.sim_base import SimRolloutBase
from sbto.tasks.task_base import OCPBase
from sbto.solvers.solver_base import SamplingBasedSolver, SolverState
from sbto.utils.modulation import step_mod, step_mod_transition

Array = npt.NDArray[np.float64]

def compute_cost(
    u_knots,
    sim: SimRolloutBase,
    task: OCPBase,
    **kwargs
    ):
    return task.cost(*sim.rollout(u_knots)[1:])

def compute_cost_t_end(
    u_knots,
    sim: SimRolloutBase,
    task: OCPBase,
    t_end: int,
    **kwargs
    ):
    # Rescale cost based on the number of timesteps
    scale = sim.T / t_end
    return scale * task.cost(*sim.rollout_t_steps(u_knots, t_end)[1:])

def compute_cost_multiple_shooting(
    u_knots,
    sim: SimRolloutBase,
    task: OCPBase,
    **kwargs
    ):
    x_shooting = task.ref.x[sim.t_knots]
    return task.cost(*sim.rollout_multiple_shooting(u_knots, x_shooting)[1:])

def _optimize(
    sim: SimRolloutBase,
    task: OCPBase,
    solver: SamplingBasedSolver,
    compute_cost_fn: Any,
    init_state_solver: Optional[SolverState] = None,
    ) -> Tuple[SolverState, Array, Array]:
    all_costs = []
    all_samples = []
    pbar = trange(solver.cfg.N_it, desc="Optimizing", leave=True)
    pbar_postfix = {}

    if not init_state_solver is None:
        solver.state = copy.deepcopy(init_state_solver)

    start = time.time()
    for _ in pbar:
        samples = solver.get_samples()
        all_samples.append(samples.copy())

        costs = compute_cost_fn(samples, sim, task)
        all_costs.append(costs)

        solver.update(samples, costs)

        pbar_postfix["min_cost"] = solver.state.min_cost_all
        pbar_postfix["cost"] = solver.state.min_cost

        pbar.set_postfix(pbar_postfix)

    end = time.time()
    duration = end - start
    print(f"Solving time: {duration:.2f}s")

    all_samples_arr = np.asarray(all_samples)
    all_costs_arr = np.asarray(all_costs)
    last_solver_state = copy.deepcopy(solver.state)

    return last_solver_state, all_samples_arr, all_costs_arr

def optimize_single_shooting(
    sim: SimRolloutBase,
    task: OCPBase,
    solver: SamplingBasedSolver,
    init_state_solver: Optional[SolverState] = None 
    ) -> Tuple[SolverState, Array, Array]:
    return _optimize(
        sim,
        task,
        solver,
        compute_cost,
        init_state_solver, 
    )

def optimize_mutiple_shooting(
    sim: SimRolloutBase,
    task: OCPBase,
    solver: SamplingBasedSolver,
    init_state_solver: Optional[SolverState] = None 
    ) -> Tuple[SolverState, Array, Array]:
    return _optimize(
        sim,
        task,
        solver,
        compute_cost_multiple_shooting,
        init_state_solver,
    )

def optimize_incremental_opt(
    sim: SimRolloutBase,
    task: OCPBase,
    solver: SamplingBasedSolver,
    init_state_solver: Optional[SolverState] = None,
    N_max_it_per_knots: int = 50,
    min_std_next: float = 1.e-2,
    ) -> Tuple[SolverState, Array, Array]:
    all_costs = []
    all_samples = []

    if not init_state_solver is None:
        solver.state = copy.deepcopy(init_state_solver)

    start = time.time()

    reset_best_knots_all = True
    pbar_knots = trange(sim.Nknots-1, desc="Optimizing", leave=True)
    pbar_postfix = {}

    for N_knots_to_opt in pbar_knots:
        N_knots_to_opt += 1  # starts with first 2 knots
        nit = 0
        max_std_diag = np.inf
        pbar_knots.set_description_str(f"Opt. first {N_knots_to_opt+1} knots")
        pbar_it = trange(N_max_it_per_knots, leave=False)

        t_end = sim.t_knots[N_knots_to_opt] + 1
        N_var_to_opt = (N_knots_to_opt + 1) * sim.Nu
        solver.opt_first_dim(N_var_to_opt)

        all_knots_optimized = N_knots_to_opt == sim.Nknots-1
        while min_std_next < max_std_diag and nit < N_max_it_per_knots:
            # Reset best knots when all knots are optimized
            if reset_best_knots_all and all_knots_optimized:
                solver.state.min_cost_all = np.inf
                reset_best_knots_all = False

            samples = solver.get_samples()
            if all_knots_optimized:
                all_samples.append(samples.copy())

            costs = compute_cost_t_end(samples, sim, task, t_end=t_end)
            all_costs.append(costs)

            solver.update(samples, costs)

            max_std_diag = np.max(np.diag(solver.state.cov)[:N_var_to_opt])
            nit += 1

            pbar_it.update(1)
            pbar_postfix["min_cost"] = solver.state.min_cost_all
            pbar_postfix["cost"] = solver.state.min_cost
            pbar_postfix["std_max"] = max_std_diag
            pbar_knots.set_postfix(pbar_postfix)
        
        del pbar_it

    end = time.time()
    duration = end - start
    print(f"Solving time: {duration:.2f}s")

    all_samples_arr = np.asarray(all_samples)
    all_costs_arr = np.asarray(all_costs)
    last_solver_state = copy.deepcopy(solver.state)

    return last_solver_state, all_samples_arr, all_costs_arr
