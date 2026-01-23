#!/usr/bin/env python3
import sys
import os
import shlex
import subprocess
import hydra
import datetime
import pathlib
import tempfile

SWEEP_FLAGS = {"-m", "--multirun"}

def split_hydra_args(argv):
    """
    - sweep_triggers: flags that cause Hydra sweep in job.py (-m, --multirun, x=1,2)
    - hydra_args: all original Hydra overrides preserved
    - clean_argv: what Hydra in job.py should see (no sweep flags)
    """
    sweep_triggers = []
    clean_argv = []
    hydra_args = []  # original, complete hydra args passed by user

    for a in argv:
        hydra_args.append(a)

        # direct sweep flags
        if a in SWEEP_FLAGS:
            sweep_triggers.append(a)
            continue

        # sweep syntax (e.g. lr=1e-3,1e-4)
        if "=" in a and "," in a:
            sweep_triggers.append(a)
            continue

        clean_argv.append(a)

    return clean_argv, hydra_args, sweep_triggers


# --- Strip sweep triggers BEFORE Hydra sees them ---
clean_argv, hydra_args_full, sweep_triggers = split_hydra_args(sys.argv[1:])
sys.argv = [sys.argv[0]] + clean_argv
# ---------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    hydra_rundir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # -------------------------
    # Freeze code at git commit
    # -------------------------
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

    scratch_root = hydra_rundir
    os.makedirs(hydra_rundir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(scratch_root, f"job_{timestamp}_{commit[:8]}")

    # Create immutable snapshot via git archive (faster than clone)
    subprocess.check_call(["mkdir", "-p", run_dir])
    tar_path = os.path.join(scratch_root, f"{commit}.tar")
    subprocess.check_call(["git", "archive", commit, "--format=tar", "--output", tar_path])
    subprocess.check_call(["tar", "-xf", tar_path, "-C", run_dir])

    # Log commit
    with open(os.path.join(run_dir, "GIT_COMMIT.txt"), "w") as f:
        f.write(commit + "\n")

    print(f"[INFO] Code frozen at commit {commit}")
    print(f"[INFO] Run directory: {run_dir}")

    # -------------------------
    # Build Slurm arguments
    # -------------------------
    slurm = cfg.slurm
    slurm_args = [
        f"--partition={slurm.partition}",
        f"--clusters={slurm.clusters}",
        f"--nodes={slurm.nodes}",
        f"--cpus-per-task={slurm.cpus}",
        f"--time={slurm.time}",
        f"--mem={slurm.mem}",
        f"--job-name={slurm.jobname}",
        f"--output={run_dir}/slurm-%j.out",
        f"--chdir={run_dir}",
    ]

    # Python entry point inside frozen snapshot
    python_script = os.path.join(run_dir, "sbto/main.py")
    quoted_script = shlex.quote(python_script)

    # Forward Hydra args (reinsert sweep mode if detected)
    if sweep_triggers:
        hydra_forward = ["-m"] + hydra_args_full
    else:
        hydra_forward = hydra_args_full

    quoted_args = " ".join(shlex.quote(a) for a in hydra_forward)

    # -------------------------
    # Conda activation (robust)
    # -------------------------
    conda_activate = (
        "source /dss/lrzsys/sys/spack/release/24.4.0/opt/x86_64/miniconda3/"
        "24.7.1-gcc-t6x7erm/bin/activate sbto"
    )

    wrapped_cmd = f"bash -lc '{conda_activate} && python3 {quoted_script} {quoted_args}'"

    cmd = ["sbatch", f"--wrap={wrapped_cmd}"] + slurm_args

    print("Submitting job:")
    print(" ".join(shlex.quote(c) for c in cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
