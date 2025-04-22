import numpy as np

def laplacian(field, dx):
    """
    Compute discrete Laplacian of a 2D or 1D field with Neumann BCs.
    """
    if field.ndim == 1:
        lap = np.zeros_like(field)
        # simple second-order finite difference
        lap[1:-1] = (field[2:] - 2*field[1:-1] + field[:-2]) / dx**2
        lap[0] = lap[1]
        lap[-1] = lap[-2]
        return lap
    elif field.ndim == 2:
        lap = np.zeros_like(field)
        lap[1:-1,1:-1] = (
            field[2:,1:-1] + field[:-2,1:-1] +
            field[1:-1,2:] + field[1:-1,:-2] -
            4*field[1:-1,1:-1]
        ) / dx**2
        lap[0,:]   = lap[1,:]
        lap[-1,:]  = lap[-2,:]
        lap[:,0]   = lap[:,1]
        lap[:,-1]  = lap[:,-2]
        return lap
    else:
        raise ValueError("Field must be 1D or 2D")