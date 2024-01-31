import numpy as np
from numpy.linalg import norm
from typing import Union


# Función de optimización basada en referencia: Direct solution of Schrödinger equation by GA
def compute_z(
    x: Union[np.array, list], energy_lowbound=0, well_length=5, force_boundary=True
) -> float:
    # Cost goes to infinity if boundary conditions are not satisfied
    if force_boundary:
        if abs(x[0]) > 0.01:
            return 9e9 * abs(x[0])
        if abs(x[-1]) > 0.01:
            return 9e9 * abs(x[-1])

    interval_size = well_length / (len(x) - 1)

    # First step (second derivative is assumed to be that of next point)
    energy_exp = -0.5 * (x[0] * (x[0] + x[2] - 2 * x[1])) / interval_size
    # Accounts for only half an interval
    j = (x[0] ** 2) * interval_size / 2

    # Middle steps
    for i in range(1, len(x) - 1):
        energy_exp += -0.5 * (x[i] * (x[i - 1] + x[i + 1] - 2 * x[i])) / interval_size
        j += (x[i] ** 2) * interval_size

    # Final step (second derivative is assumed to be that of previous point)
    energy_exp += -0.5 * (x[-1] * (x[-1] + x[-3] - 2 * x[-2])) / interval_size
    # Accounts for only half an interval
    j += (x[-1] ** 2) * interval_size / 2

    Z = (energy_exp / j - energy_lowbound) ** 2
    return Z