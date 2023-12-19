import warnings
import numpy as np


def swap(src, dst, arrays, nu, p):
    for n in range(len(arrays)):
        if arrays[n] is not None:
            arrays[n][*src], arrays[n][*dst] = arrays[n][*dst], arrays[n][*src]
    nu[src[0], src[1]] += 1.0 / p.nm
    nu[dst[0], dst[1]] -= 1.0 / p.nm
    return [arrays, nu]


def get_solid_fraction(s: np.ndarray, loc: list | None = None) -> float:
    """Calculate solid fraction of a single physical in a 3D array.

    Args:
        s: a 3D numpy array
        loc: Either None or a list of two integers.

    Returns:
        The fraction of the solid phase in s at (i, j) as a float.
    """
    # return np.mean(~np.isnan(s[i, j, :]))
    if loc is None:
        return 1.0 - np.mean(np.isnan(s), axis=2)
    else:
        return 1.0 - np.mean(np.isnan(s[loc[0], loc[1], :]))


def get_average(s, loc: list | None = None):
    """
    Calculate the mean size over the microstructural co-ordinate.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if loc is None:
            s_bar = np.nanmean(s, axis=2)
        else:
            s_bar = np.nanmean(s[loc[0], loc[1], :])
    return s_bar


def get_hyperbolic_average(s: np.ndarray, loc: list | None = None) -> float:
    """
    Calculate the hyperbolic mean size over the microstructural co-ordinate.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if loc is None:
            return 1.0 / np.nanmean(1.0 / s, axis=2)
        else:
            return 1.0 / np.nanmean(1.0 / s[loc[0], loc[1], :])


def get_depth(s):
    """
    Unused.
    """
    depth = np.mean(np.mean(~np.isnan(s), axis=2), axis=1)
    return depth
