import numpy as np
from utils.math_utils import laplacian


import numpy as np

def sim_buckling(E, nu, t_c, P, q, x_domain):
    """
    Solve
      d^2/dx^2 [ (E/(1-nu^2)) * t_c(x)^3/12 * d^2w/dx^2 ]
      + P*t_c(x) * d^2w/dx^2 = q
    on the grid x_domain with w(0)=w(L)=0.
    
    Parameters
    ----------
    E : float or array of length N
    nu: float
    t_c : array of length N
    P : float
    q : float or array of length N
    x_domain : array of length N
    
    Returns
    -------
    w : array of length N
        deflection profile
    """
    N = x_domain.size
    dx = x_domain[1] - x_domain[0]
    
    # coefficient functions
    a = (E / (1 - nu**2)) * (t_c**3) / 12.0   # shape (N,)
    b = P * t_c                              # shape (N,)
    q_vec = np.full(N, q) if np.isscalar(q) else q.copy()
    
    # build 2nd‐derivative matrix D2 (central diffs, Dirichlet at ends)
    D2 = np.zeros((N, N))
    for i in range(1, N-1):
        D2[i, i-1] =  1.0
        D2[i, i  ] = -2.0
        D2[i, i+1] =  1.0
    D2 /= dx**2
    
    # assemble operator: L = D2 @ diag(a) @ D2 + diag(b) @ D2
    A = np.diag(a)
    B = np.diag(b)
    L = D2.dot(A.dot(D2)) + B.dot(D2)
    
    # impose w(0)=w(L)=0 (Dirichlet BC)
    L[0, :] = 0.0
    L[0, 0] = 1.0
    q_vec[0] = 0.0
    
    L[-1, :] = 0.0
    L[-1, -1] = 1.0
    q_vec[-1] = 0.0
    
    # solve linear system
    w = np.linalg.solve(L, q_vec)
    return w