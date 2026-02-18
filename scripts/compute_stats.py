import argparse
import shutil
import numpy as np
from sbto.evaluation.load import load_dataset_with_errors as load

def main(
    task_dir: str,
    delete_failures: bool = False,
    ):
    data = load(task_dir)

    if delete_failures:
        failure_rundirs = data[~data["success"]]["rundir"]
        for rundir in failure_rundirs:
            shutil.rmtree(rundir)
        print(len(failure_rundirs), "failure runs deleted.")

    print("=== Success Counts ===")
    n_success = data["success"].sum()
    print(f" {n_success / len(data) * 100.}%")

    print("=== Smoothness ===")
    smoothness = data[data["success"]]["act_acc_ratio"].values.mean()
    print(f" {smoothness:.2f}")

    print("=== Compute ===")
    T = data[data["success"]]["T"].values
    dt = data[data["success"]]["dt"].values
    duration = T * dt
    sim_steps = data[data["success"]]["total_sim_timesteps"].values
    opt_duration = data[data["success"]]["opt_duration"].values
    
    avg_sim_step_per_sec_success = np.mean(sim_steps / duration)
    avg_opt_duration_per_sec_success = np.mean(opt_duration / duration)
    print(f" Sim step per second of motion: {avg_sim_step_per_sec_success:.2f} (sim steps/s)")
    print(f" Optimization time per second of motion: {avg_opt_duration_per_sec_success:.2f} (s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate SBTO dataset statistics."
    )

    parser.add_argument(
        "task_dir",
        type=str,
        help="Path to the dataset root directory",
    )

    parser.add_argument(
        "--delete-failures",
        action="store_true",
        help="Delete failed run directories",
    )

    args = parser.parse_args()

    main(
        task_dir=args.task_dir,
        delete_failures=args.delete_failures,
    )