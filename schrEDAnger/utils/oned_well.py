import numpy as np
from numpy.linalg import norm
from typing import Union


# Obtiene la energía del n-ésimo estado del pozo infinito 1D
def get_nth_state_energy(n_state: int, well_length=5):
    if not isinstance(n_state, int):
        raise ValueError("States must be quantized as integers")
    ground_energy = (2 * np.pi) ** 2 / (8 * well_length**2)
    if n_state > 0:
        nth_state_energy = (n_state + 1) ** 2 * ground_energy
        return nth_state_energy
    else:
        return ground_energy


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

    ### INTEGRAL CALCULATION ###

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

    ### INTEGRAL CALCULATION ###

    Z = (energy_exp / j - energy_lowbound) ** 2
    return Z


# Función de optimización basada en referencia: Direct solution of Schrödinger equation by GA
def compute_z_triangles(
    psi: Union[np.array, list], energy_lowbound=0, well_length=5, force_boundary=True
) -> float:
    # Cost goes to infinity if boundary conditions are not satisfied
    if force_boundary:
        if abs(psi[0]) > 0.01:
            return 9e9 * abs(psi[0])
        if abs(psi[-1]) > 0.01:
            return 9e9 * abs(psi[-1])

    interval_size = well_length / (len(psi) - 1)


    ### INTEGRAL CALCULATION ###

    # # First step (second derivative is computed setting terms out of the array to zero)
    # energy_exp = (-1/(4*interval_size))*((psi[0] * (psi[1] - 2 * psi[0]))+(psi[1] * (psi[0] + psi[2] - 2 * psi[1])))
    # j = (psi[0] ** 2 + psi[1] ** 2) * interval_size / 2

    energy_exp = (-1/(4*interval_size))*((psi[0] * (psi[0] - 2 * psi[1] + psi[2]))+(psi[1] * (psi[0] + psi[2] - 2 * psi[1])))
    j = (psi[0] ** 2 + psi[1] ** 2) * interval_size / 2


    # Middle steps
    for i in range(1, len(psi)-3): #antes -2
        energy_exp += (-1/(4*interval_size))*((psi[i] * (psi[i - 1] + psi[i + 1] - 2 * psi[i]))+(psi[i+1] * (psi[i] + psi[i+2] - 2 * psi[i+1])))
        j += (psi[i] ** 2 + psi[i + 1] ** 2) * interval_size / 2
        
    # # Final step (second derivative is computed setting terms out of the array to zero)
    # energy_exp += (1/(4*interval_size))*((psi[-2] * (psi[-3] + psi[-1] - 2 * psi[-2]))+(psi[-1] * (psi[-2] - 2 * psi[-1])))
    # j += (psi[-2] ** 2 + psi[-1] ** 2) * interval_size / 2
    
    ##outside loop terms (outside box psi=0)
    energy_exp += (-1/(4*interval_size))*((psi[-2] * (psi[-3] + psi[-1] - 2 * psi[-2]))+(psi[-1] * (psi[-3] - 2 * psi[-2] + psi[-1])))
    j += (psi[-2] ** 2 + psi[-1] ** 2) * interval_size / 2

    energy_exp += (-1/(4*interval_size))*psi[-1]*(psi[-3]+psi[-1]- 2 * psi[-2])
    j += psi[-1]**2 * interval_size / 2 

    ### INTEGRAL CALCULATION ###

    Z = (energy_exp / j - energy_lowbound) ** 2
    return Z
