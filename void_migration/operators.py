import warnings
import numpy as np


def swap(src, dst, arrays, nu, p):
    for n in range(len(arrays)):
        if arrays[n] is not None:
            arrays[n][*src], arrays[n][*dst] = arrays[n][*dst], arrays[n][*src]
    nu[src[0], src[1]] += 1.0 / p.nm
    nu[dst[0], dst[1]] -= 1.0 / p.nm
    return [arrays, nu]


def get_solid_fraction(s: np.ndarray, i: int, j: int) -> float:
    """Calculate solid fraction of a single physical in a 3D array.

    Args:
        s: a 3D numpy array
        i: an integer representing a row index
        j: an integer representing a column index

    Returns:
        The fraction of the solid phase in s at (i, j) as a float.
    """
    # return np.mean(~np.isnan(s[i, j, :]))
    return 1.0 - np.mean(np.isnan(s[i, j, :]))


def get_average(s):
    """
    Calculate the mean size over the microstructural co-ordinate.
    Then calculate along every row
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        s_bar = np.nanmean(np.nanmean(s, 2), 0)
    return s_bar


def get_hyperbolic_average(s):
    """
    Calculate the hyperbolic mean size over the microstructural co-ordinate.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        s_inv_bar = 1.0 / np.nanmean(1.0 / s, 2)
    return s_inv_bar


def get_depth(s):
    """
    Unused.

    den = 1 - np.mean(np.isnan(s), axis=2)
    ht = []
    for w in range(p.nx):
        if np.mean(den[w]) > 0:
            ht.append(np.max(np.nonzero(den[w])))
        else:
            ht.append(np.argmin(den[w]))
    """
    depth = np.mean(np.mean(~np.isnan(s), axis=2), axis=1)
    return depth
