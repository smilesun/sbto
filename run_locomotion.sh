# uv run sbto/main.py task=g1/robot_ref task.cfg_ref.motion_path=datasets/robot-object/sub10_largebox_000_original.npz solver=cem 
uv run sbto/main.py task=g1/robot_ref task.cfg_ref.motion_path=datasets/robot-object/sub10_largebox_000_original.npz solver=cbo solver.cfg.lambda_=10 solver.cfg.min_it_per_knot=100 warm_start.start_knots=2 warm_start.N_max_incr=1000 solver.cfg.N_samples=4096
