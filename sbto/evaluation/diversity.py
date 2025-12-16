import numpy as np

def avg_joint_variance(joint_traj):
    """
    joint_traj: [T, N] or [B, T, N]
    """
    traj = np.asarray(joint_traj)
    
    if traj.ndim == 2:
        traj = traj[None, ...]
    
    std_knots_timestep = np.std(traj, axis=0)
    std_avg_over_knots = np.mean(std_knots_timestep, axis=-1)
    std_avg_over_time = np.mean(std_knots_timestep, axis=0)
    std_avg = np.mean(std_knots_timestep)

    # If original input was unbatched, return scalar
    if std_avg_over_time.shape[0] == 1:
        return std_avg, std_avg_over_time[0], std_avg_over_knots
    
    return std_avg, std_avg_over_time, std_avg_over_knots