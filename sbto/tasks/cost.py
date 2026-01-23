import numpy as np
import numpy.typing as npt
from numba import njit, prange

Array = npt.NDArray[np.float64]
    
@njit(parallel=True, fastmath=True, cache=True)
def quadratic_cost_nb(var, ref, weight):
    N, T, I = var.shape
    result = np.zeros(N, np.float64)
    for n in prange(N):
        total = 0.0
        for t in range(T):
            for i in range(I):
                diff = var[n, t, i] - ref[t, i]
                total += weight[t, i] * diff * diff
        result[n] = total
    return result
    
@njit(parallel=True, fastmath=True, cache=True)
def quaternion_dist_nb(var, ref, weights):
    """
    Numba-accelerated version of quaternion distance cost.
    Shapes:
        var: (N, T, Nquat*4)       # quaternion rollout
        ref: (T, Nquat*4)          # reference quaternion trajectory
        weights: (T, 1) or (T,)  # scalar weights per timestep
    Returns:
        cost: (N,)
    """
    N, T, Q = var.shape
    result = np.zeros(N, dtype=np.float64)
    QUAT_SIZE = 4
    Nquat = Q // QUAT_SIZE

    for n in prange(N):
        total = 0.0
        for t in range(T):
            for iquat in range(Nquat):
                dot = 0.0
                for k in range(iquat * QUAT_SIZE, (iquat+1) * QUAT_SIZE):
                    dot += var[n, t, k] * ref[t, k]
                diff = 1.0 - dot * dot
                total += weights[t, 0] * diff
        result[n] = total
    return result


@njit(fastmath=True)
def quat_mul_nb(q1, q2):
    """Multiply two quaternions (w,x,y,z)."""
    return np.array([
        q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
        q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
        q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
        q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0],
    ])


@njit(fastmath=True)
def quat_to_vel_nb(q):
    """Convert quaternion error to axis-angle vector."""
    w, x, y, z = q
    sin_a2 = np.sqrt(x*x + y*y + z*z)

    if sin_a2 < 1e-12:
        return np.zeros(3)

    angle = 2.0 * np.arctan2(sin_a2, w)

    # wrap to [-pi, pi]
    if angle > np.pi:
        angle -= 2.0 * np.pi

    inv = angle / sin_a2
    return np.array([x*inv, y*inv, z*inv])

@njit(parallel=True, fastmath=True, cache=True)
def quaternion_dist_logmap_nb(var, ref, weights):
    N, T, Q = var.shape
    QUAT_SIZE = 4
    Nquat = Q // QUAT_SIZE
    result = np.zeros(N)

    for n in prange(N):
        total = 0.0

        for t in range(T):
            wgt = weights[t, 0] if weights.ndim == 2 else weights[t]

            for i in range(Nquat):
                k = i * 4

                wa, xa, ya, za = var[n,t,k], var[n,t,k+1], var[n,t,k+2], var[n,t,k+3]
                wb, xb, yb, zb = ref[t,k], ref[t,k+1], ref[t,k+2], ref[t,k+3]

                # qb inverse
                wb, xb, yb, zb = wb, -xb, -yb, -zb

                # q_err = qb_inv * qa
                w0 = wb*wa - xb*xa - yb*ya - zb*za
                x0 = wb*xa + xb*wa + yb*za - zb*ya
                y0 = wb*ya - xb*za + yb*wa + zb*xa
                z0 = wb*za + xb*ya - yb*xa + zb*wa

                sin_a2 = np.sqrt(x0*x0 + y0*y0 + z0*z0)
                if sin_a2 > 1e-12:
                    angle = 2.0 * np.arctan2(sin_a2, w0)
                    if angle > np.pi:
                        angle -= 2.0*np.pi
                    inv = angle / sin_a2
                    total += wgt * (x0*x0 + y0*y0 + z0*z0) * (inv*inv)

        result[n] = total
    return result


@njit(parallel=True, fastmath=True, cache=True)
def hamming_dist_nb(cnt_rollout, cnt_plan, weights):
    """
    Efficient Hamming-distance-based contact cost.
    Args:
        cnt_rollout : (N, T, C) array, contact states (0, 1, maybe >1)
        cnt_plan           : (T, C) array, desired contact pattern (0 or 1)
        weights            : (T, C) array of float32 weights

    Returns:
        cost : (N,) array of float32
    """
    N, T, C = cnt_rollout.shape
    result = np.zeros(N, dtype=np.float64)

    for n in prange(N):
        total = 0.0
        for t in range(T):
            for c in range(C):
                s = cnt_rollout[n, t, c]
                # Clamp contact status > 1 to 1, cast to integer
                if s > 1:
                    s = 1
                # XOR trick for mismatch detection (works with ints 0/1)
                diff = int(s) ^ int(cnt_plan[t, c])
                total += weights[t, c] * diff
        result[n] = total
    return result
