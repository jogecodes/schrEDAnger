import numpy as np


def get_nth_state_energy(n_state: int, well_length=5):
    if not isinstance(n_state, int):
        raise ValueError("States must be quantized as integers")
    ground_energy = (2 * np.pi) ** 2 / (8 * well_length**2)
    if n_state > 0:
        nth_state_energy = (n_state + 1) ** 2 * ground_energy
        return nth_state_energy
    else:
        return ground_energy
