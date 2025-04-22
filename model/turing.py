import numpy as np
from utils.math_utils import laplacian


def sim_turing(Du, Dv, f_func, g_func, u0, v0, dx, dt, steps):
    """
    Simulate reaction-diffusion:
      du/dt = Du*Laplacian(u) + f(u,v)
      dv/dt = Dv*Laplacian(v) + g(u,v)

    Parameters
    ----------
    Du, Dv: float  # diffusion coefficients
    f_func, g_func: callables
    u0, v0: 2D arrays  # initial conditions
    dx, dt: float
    steps: int

    Returns
    -------
    u, v: 2D arrays
    """
    u, v = u0.copy(), v0.copy()
    for _ in range(steps):
        Lu = laplacian(u, dx)
        Lv = laplacian(v, dx)
        u += dt*(Du*Lu + f_func(u, v))
        v += dt*(Dv*Lv + g_func(u, v))
    return u, v