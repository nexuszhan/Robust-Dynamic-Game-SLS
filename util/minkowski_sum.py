import numpy as np
from numpy.linalg import eig

def minkowski_sum_N_ellipsoids_outer(Phis, mode="trace"):
    """
    Ps: list of (d,d) SPD matrices; cs: list of (d,) centers; all same d (e.g., d=4)
    mode: 'trace' (closed-form) or a given weight array of shape (N,) summing to 1
    Returns: center, U, semi_axes, R
    """
    Ps = np.zeros_like(Phis)
    N = Phis.shape[0]
    for i in range(N):
        Ps[i,:,:] = Phis[i] @ Phis[i].T
    # d = Ps[0].shape[0]

    if mode == "trace":
        T = np.array([np.trace(P) for P in Ps])
        s = np.sqrt(T)
        w = s / s.sum()
    else:
        w = np.asarray(mode, float)
        assert w.shape == (N,) and np.all(w > 0) and np.isclose(w.sum(), 1.0)

    R_sup = sum(Ps[i] / w[i] for i in range(N))
    R_sup = 0.5 * (R_sup + R_sup.T)  # symmetrize

    vals, vecs = eig(R_sup)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx].real, vecs[:, idx].real

    semi_axes = np.sqrt(vals)
    return vecs, semi_axes, R_sup
