import numpy as np

import multiprocessing


def _unpacking_apply_along_axis(all_args):
    """
    Like numpy.apply_along_axis(), but with arguments in a tuple
    instead.

    This function is useful with multiprocessing.Pool().map(): (1)
    map() only handles functions that take a single argument, and (2)
    this function can generally be imported from a module, as required
    by map().
    """
    (func1d, axis, arr, args, kwargs) = all_args
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)


def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [
        (func1d, effective_axis, sub_arr, args, kwargs)
        for sub_arr in np.array_split(arr, multiprocessing.cpu_count())
    ]

    pool = multiprocessing.Pool()
    individual_results = pool.map(_unpacking_apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)


def get_nth_state_energy(n_state: int, well_length=5):
    if not isinstance(n_state, int):
        raise ValueError("States must be quantized as integers")
    ground_energy = (2 * np.pi) ** 2 / (8 * well_length**2)
    if n_state > 0:
        nth_state_energy = (n_state + 1) ** 2 * ground_energy
        return nth_state_energy
    else:
        return ground_energy
