# uv run sbto/main.py solver=pcbo task.cfg_ref.motion_path=datasets/robot-object/sub10_largebox_000_original.npz
uv run sbto/main.py solver=cbo warm_start.start_knots=15 warm_start.N_max_incr=1000 solver.cfg.N_samples=2048 task.cfg_ref.motion_path=datasets/robot-object/sub10_largebox_000_original.npz
solver.cfg.min_it_per_knot=1000
