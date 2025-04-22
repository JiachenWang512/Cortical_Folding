import numpy as np
from utils.math_utils import laplacian


def sim_turing(Du, Dv, f_func, g_func, u0, v0, dx, dt, steps):
    """
    Explicit Euler with adaptive sub‑stepping for stability:
      du/dt = Du*Lap(u) + f(u,v)
      dv/dt = Dv*Lap(v) + g(u,v)
    """
    u = u0.copy()
    v = v0.copy()

    # maximum stable dt for diffusion
    dt_max = dx**2 / (4.0 * max(Du, Dv))
    # if your input dt is too big, break it into substeps
    n_sub = int(np.ceil(dt / dt_max))
    dt_sub = dt / n_sub

    for _ in range(steps):
        for _ in range(n_sub):
            Lu = laplacian(u, dx)
            Lv = laplacian(v, dx)

            u += dt_sub * (Du*Lu + f_func(u, v))
            v += dt_sub * (Dv*Lv + g_func(u, v))

            # optional: keep values in a reasonable range
            u = np.clip(u, -5, 5)
            v = np.clip(v, -5, 5)

    return u, v