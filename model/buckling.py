import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

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
    
    a = (E / (1 - nu**2)) * (t_c**3) / 12.0
    b = P * t_c                             
    q_vec = np.full(N, q) if np.isscalar(q) else q.copy()
    
    D2 = np.zeros((N, N))
    for i in range(1, N-1):
        D2[i, i-1] =  1.0
        D2[i, i  ] = -2.0
        D2[i, i+1] =  1.0
    D2 /= dx**2
    
    # L = D2 @ diag(a) @ D2 + diag(b) @ D2
    A = np.diag(a)
    B = np.diag(b)
    L = D2.dot(A.dot(D2)) + B.dot(D2)
    
    # w(0)=w(L)=0 (Dirichlet BC)
    L[0, :] = 0.0
    L[0, 0] = 1.0
    q_vec[0] = 0.0
    
    L[-1, :] = 0.0
    L[-1, -1] = 1.0
    q_vec[-1] = 0.0

    w = np.linalg.solve(L, q_vec)
    return w

def estimate_wavelength(pattern, dx):

    F = np.fft.fft2(pattern)
    P2 = np.abs(F)**2
    P2_shift = np.fft.fftshift(P2)
    freqs = np.fft.fftfreq(pattern.shape[0], d=dx)
    fx, fy = np.meshgrid(freqs, freqs)
    r = np.sqrt(fx**2 + fy**2)
    n_bins = 50
    r_flat = r.flatten(); P_flat = P2_shift.flatten()
    bins = np.linspace(0, r.max(), n_bins)
    bin_idx = np.digitize(r_flat, bins)
    mag = [P_flat[bin_idx==i].mean() for i in range(1,len(bins))]
    peak = np.argmax(mag)
    return 1.0 / bins[peak]


def sim_buckling_2d(E, nu, tc, P, q, Lx, Ly):
    """
    Solve 2D buckling with w=0 on all boundaries.
    Inputs:
      E, nu    : material constants
      tc       : (Ny,Nx) array of thickness
      P        : compressive load magnitude
      q        : (Ny,Nx) forcing term
      Lx, Ly   : domain size
    Returns:
      w        : (Ny,Nx) deflection field
    """
    Ny, Nx = tc.shape
    dx = Lx/(Nx-1); dy = Ly/(Ny-1)
    N = Nx*Ny

    ex = np.ones(Nx); ey = np.ones(Ny)
    D2x = sp.diags([ex, -2*ex, ex], [-1,0,1], shape=(Nx,Nx)) / dx**2
    D2y = sp.diags([ey, -2*ey, ey], [-1,0,1], shape=(Ny,Ny)) / dy**2

    L2 = sp.kron(sp.eye(Ny), D2x) + sp.kron(D2y, sp.eye(Nx))

    D_flat = (E/(1 - nu**2)) * (tc.ravel()**3) / 12.0
    B_flat = P * tc.ravel()
    D_mat  = sp.diags(D_flat, 0)
    B_mat  = sp.diags(B_flat, 0)

    A = L2.dot(D_mat.dot(L2)) + B_mat.dot(L2)
    q_arr = np.asarray(q)
    if q_arr.shape == ():
        b = q_arr * np.ones(N)
    else:
        if q_arr.shape != (Ny, Nx):
            raise ValueError(f"q must have shape {(Ny,Nx)}, but got {q_arr.shape}")
        b = q_arr.ravel()

    A = A.tolil()
    for j in range(Ny):
        for i in (0, Nx-1):
            idx = j*Nx + i
            A.rows[idx] = [idx]; A.data[idx] = [1.0]; b[idx] = 0.0
    for i in range(Nx):
        for j in (0, Ny-1):
            idx = j*Nx + i
            A.rows[idx] = [idx]; A.data[idx] = [1.0]; b[idx] = 0.0

    A = A.tocsc()
    w_flat = spla.spsolve(A, b)
    return w_flat.reshape(Ny, Nx)

def estimate_wavelength_2d(pattern, dx):
    """
    Return (lamb_radial, lamb_x, lamb_y, (f_x, f_y)) for the strongest nonzero 2D mode.
    """

    P = pattern - np.mean(pattern)

    F   = np.fft.fft2(P)
    P2  = np.abs(F)**2
    P2s = np.fft.fftshift(P2)

    ny, nx = pattern.shape
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    FX, FY = np.meshgrid(np.fft.fftshift(fx),
                         np.fft.fftshift(fy))

    cx, cy = nx//2, ny//2
    P2s[cy, :] = 0    # remove f_y=0 line
    P2s[:, cx] = 0    # remove f_x=0 line

    idx   = np.unravel_index(np.argmax(P2s), P2s.shape)
    fxp   = FX[idx]; fyp = FY[idx]
    fmag  = np.hypot(fxp, fyp)

    lamb_radial = 1/fmag   if fmag!=0 else np.nan
    lamb_x      = 1/abs(fxp) if fxp!=0 else np.nan
    lamb_y      = 1/abs(fyp) if fyp!=0 else np.nan

    return lamb_radial, lamb_x, lamb_y, (fxp, fyp)