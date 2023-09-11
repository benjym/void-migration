#!/usr/bin/python

import plotter

__doc__ = """
void_migration.py

This script simulates ....
"""
__author__ = "Benjy Marks"
__version__ = "0.1"

import os
import sys
import numpy as np
from tqdm import tqdm
import warnings
import json5
from itertools import product
from numba import jit, njit

from numpy.typing import ArrayLike

# warnings.filterwarnings("ignore")


class dict_to_class:
    """
    A convenience class to store the information from the parameters dictionary. Used because I prefer using p.variable to p['variable'].
    """

    def __init__(self, dict: dict):
        list_keys: List[str] = []
        lists: List[List] = []
        for key in dict:
            setattr(self, key, dict[key])
            if isinstance(dict[key], list):
                list_keys.append(key)
                lists.append(dict[key])
        setattr(self, "lists", lists)
        setattr(self, "list_keys", list_keys)


def IC(p):
    """
    Sets up the initial value of the grain size and/or void distribution everywhere.

    Args:
        p: Parameters class. In particular, the `gsd_mode` and `IC_mode` should be set to determine the grain size distribution (gsd) and the initial condition (IC).

    Returns:
        The array of grain sizes. Values of `NaN` are voids.
    """

    # pick a grain size distribution
    if p.gsd_mode == "mono":
        s = np.ones([p.nx, p.ny, p.nm])  # monodisperse
        p.s_M = p.s_m
    if p.gsd_mode == "bi":  # bidisperse
        s = np.random.choice([p.s_m, p.s_M], size=[p.nx, p.ny, p.nm])
    elif p.gsd_mode == "poly":  # polydisperse
        # s_0 = p.s_m / (1.0 - p.s_m)  # intermediate calculation
        s_non_dim = np.random.rand(p.nx, p.ny, p.nm)
        # s = (s + s_0) / (s_0 + 1.0)  # now between s_m and 1
        s = (p.s_M - p.s_m) * s_non_dim + p.s_m

    # where particles are in space
    if p.IC_mode == "random":  # voids everywhere randomly
        mask = np.random.rand(p.nx, p.ny, p.nm) > p.fill_ratio
    elif p.IC_mode == "top":  # voids at the top
        mask = np.zeros([p.nx, p.ny, p.nm], dtype=bool)
        mask[:, int(p.fill_ratio * p.ny) :, :] = True
    elif p.IC_mode == "full":  # completely full
        mask = np.zeros_like(s, dtype=bool)
    elif p.IC_mode == "column":  # just middle full to top
        mask = np.ones([p.nx, p.ny, p.nm], dtype=bool)
        mask[
            p.nx // 2 - int(p.fill_ratio / 2 * p.nx) : p.nx // 2 + int(p.fill_ratio / 2 * p.nx), :, :
        ] = False
    elif p.IC_mode == "empty":  # completely empty
        mask = np.ones([p.nx, p.ny, p.nm], dtype=bool)

    s[mask] = np.nan

    return s


def move_voids_adv(u, v, s, c, T, p):  # Advection
    """
    Deprecated function. Do not use.
    """
    for j in range(p.ny - 2, -1, -1):
        x_loop = np.arange(p.nx)
        if p.internal_geometry:
            x_loop = x_loop[~p.boundary[:, j]]  # don't move apparent voids at boundaries
        np.random.shuffle(x_loop)
        for i in x_loop:
            for k in range(p.nm):
                if np.isnan(s[i, j, k]):
                    if np.random.rand() < p.P_adv:
                        if not np.isnan(s[i, j + 1, k]):
                            # if p.internal_geometry:
                            # if not p.boundary[i, j + 1]:
                            s[i, j, k], s[i, j + 1, k] = s[i, j + 1, k], s[i, j, k]
                            v[i, j] += 1
                            if T is not None:
                                T[i, j, k], T[i, j + 1, k] = T[i, j + 1, k], T[i, j, k]
    return u, v, s, c, T


def move_voids_diff(u, v, s, c, T, p):  # Diffusion
    """
    Deprecated function. Do not use.
    """
    for j in range(p.ny - 2, -1, -1):
        x_loop = np.arange(p.nx)
        if p.internal_geometry:
            x_loop = x_loop[~p.boundary[:, j]]  # don't move apparent voids at boundaries
        np.random.shuffle(x_loop)
        for i in x_loop:
            for k in range(p.nm):
                if np.isnan(s[i, j, k]):
                    if np.random.rand() < p.P_diff:
                        # CAN MAKE SEG BY SETTING PROBABILITY OF lr TO DEPEND ON SIZE RATIO
                        # lr = np.random.rand() < 1./(1. + s[i-1,j,k]/s[i+1,j,k] ) # just segregation
                        # lr = np.random.rand() < 0.5*(1 - np.sin(radians(theta))) # just slope
                        if i == 0:
                            if p.cyclic_BC:
                                lr = np.random.rand() < (1 - np.sin(np.radians(p.theta))) / (
                                    1
                                    - np.sin(np.radians(p.theta))
                                    + (1 + np.sin(np.radians(p.theta))) * (s[-1, j, k] / s[1, j, k])
                                )  # both together
                                lr = 2 * lr - 1  # rescale to +/- 1
                            else:
                                lr = 1
                        elif i == p.nx - 1:
                            if p.cyclic_BC:
                                lr = np.random.rand() < (1 - np.sin(np.radians(p.theta))) / (
                                    1
                                    - np.sin(np.radians(p.theta))
                                    + (1 + np.sin(np.radians(p.theta))) * (s[-2, j, k] / s[0, j, k])
                                )  # both together
                                lr = 2 * lr - 1  # rescale to +/- 1
                            else:
                                lr = -1
                        else:
                            if p.boundary[i - 1, j]:
                                l = 1e10  # zero chance of moving there
                            else:
                                l = s[i - 1, j, k]
                            if p.boundary[i + 1, j]:
                                r = 1e10  # zero chance of moving there
                            else:
                                r = s[i + 1, j, k]

                            lr = np.random.rand() < (1 - np.sin(np.radians(p.theta))) / (
                                1 - np.sin(np.radians(p.theta)) + (1 + np.sin(np.radians(p.theta))) * (l / r)
                            )  # both together
                            # print(
                            #     s[i - 1, j, k],
                            #     s[i, j, k],
                            #     s[i + 1, j, k],
                            #     (1 - np.sin(np.radians(p.theta)))
                            #     / (1 - np.sin(np.radians(p.theta)) + (1 + np.sin(np.radians(p.theta))) * (l / r)),
                            # )
                            lr = 2 * lr - 1  # rescale to +/- 1

                        if i == p.nx - 1 and lr == 1:  # right boundary
                            if p.cyclic_BC:
                                # if not np.isnan(s[0,j+1,k]): # this sets the angle of repose?
                                if not np.isnan(s[0, j, k]):
                                    s[-1, j, k], s[0, j, k] = s[0, j, k], s[-1, j, k]
                                    u[i, j] -= lr
                                    if T is not None:
                                        T[-1, j, k], T[0, j, k] = T[0, j, k], T[-1, j, k]
                        elif i == 0 and lr == -1:  # left boundary
                            if p.cyclic_BC:
                                # if not np.isnan(s[i+lr,j+1,k]): # this sets the angle of repose?
                                if not np.isnan(s[-1, j, k]):
                                    s[0, j, k], s[-1, j, k] = s[-1, j, k], s[0, j, k]
                                    u[i, j] -= lr
                                    if T is not None:
                                        T[0, j, k], T[-1, j, k] = T[-1, j, k], T[0, j, k]
                        else:
                            if not np.isnan(s[i + lr, j, k]):
                                print(
                                    l,
                                    r,
                                    s[i - 1, j, k],
                                    s[i, j, k],
                                    s[i + 1, j, k],
                                    (1 - np.sin(np.radians(p.theta)))
                                    / (
                                        1
                                        - np.sin(np.radians(p.theta))
                                        + (1 + np.sin(np.radians(p.theta))) * (l / r)
                                    ),
                                )
                                # if not np.isnan(s[i+lr,j+1,k]): # this sets the angle of repose at 45
                                # if np.mean(np.isnan(s[i, j, :])) > 0.5:  # if here is mostly empty (ie outside mass)
                                # print('outside')
                                if p.mu < 1:
                                    A = p.mu / 2.0  # proportion of cell that should be filled diagonally up
                                else:
                                    A = 1.0 - 1.0 / (2.0 * p.mu)
                                if (
                                    np.mean(~np.isnan(s[i + lr, j + 1, :])) > A
                                ):  # this sets an angle of repose?
                                    s[i, j, k], s[i + lr, j, k] = s[i + lr, j, k], s[i, j, k]
                                    u[i, j] -= lr
                                    if T is not None:
                                        T[i, j, k], T[i + lr, j, k] = T[i + lr, j, k], T[i, j, k]
                                # else:  # not more than 50% voids, inside mass
                                #     # print('inside')
                                #     # if p.internal_geometry and not boundary[i + lr, j]:
                                #     # if not boundary[i + lr, j]:
                                #     s[i, j, k], s[i + lr, j, k] = s[i + lr, j, k], s[i, j, k]
                                #     u[i, j] -= lr
                                #     if T is not None:
                                #         T[i, j, k], T[i + lr, j, k] = T[i + lr, j, k], T[i, j, k]

    return u, v, s, c, T


def density(s: np.ndarray, i: int, j: int) -> float:
    """Calculate density of a single physical in a 3D array.

    Args:
        s: a 3D numpy array
        i: an integer representing a row index
        j: an integer representing a column index

    Returns:
        The density of the solid phase in s at (i, j) as a float.
    """
    # return np.mean(~np.isnan(s[i, j, :]))
    return 1.0 - np.mean(np.isnan(s[i, j, :]))


# @njit
def move_voids(
    u: ArrayLike,
    v: ArrayLike,
    s: ArrayLike,
    p: dict_to_class,
    diag: int = 0,
    c: None | ArrayLike = None,
    T: None | ArrayLike = None,
) -> tuple[ArrayLike, ArrayLike, ArrayLike, None | ArrayLike, None | ArrayLike]:
    """
    Function to move voids each timestep.

    Args:
        u: Storage container for counting how many voids moved horizontally
        v: Storage container for counting how many voids moved vertically
        s: 3D array containing the local sizes everywhere. `NaN`s represent voids. Other values represent the grain size. The first two dimensions represent real space, the third dimension represents the micro-structural coordinate.
        diag: Should the voids swap horizontally (von neumnann neighbourhood, `diag=0`) or diagonally upwards (moore neighbourhood, `diag=1`). Default value `0`.
        c: If ArrayLike, a storage container for tracking motion of differently labelled particles. If `None`, do nothing.
        T: If ArrayLike, the temperature field. If `None`, do nothing.
        boundary: If ArrayLike, a descriptor of cells which voids cannot move into (i.e. boundaries). If `internal_boundary` is defined in the params file, allow for reduced movement rates rather than zero. If `None`, do nothing.

    Returns:
        u: The updated horizontal velocity
        v: The updated vertical velocity
        s: The new locations of the grains
        c: The updated concentration field
        T: The updated temperature field
    """

    # A is the proportion of the cell that should be filled diagonally up
    # Comes from a simple geometric argument where the slope goes through one corner of the cell
    if p.mu < 1:
        A = p.mu / 2.0  # proportion of cell that should be filled where the slope goes through one corner
    else:
        A = 1.0 - 1.0 / (2.0 * p.mu)

    # print(f'A = {A}')

    for j in range(p.ny - 2, -1, -1):
        if p.internal_geometry:
            x_loop = np.arange(p.nx)[~p.boundary[:, j]]  # don't move apparent voids at boundaries
        else:
            x_loop = np.arange(p.nx)
        np.random.shuffle(x_loop)
        for i in x_loop:
            if density(s, i, j) < p.critical_density:
                for k in range(p.nm):
                    if np.isnan(s[i, j, k]):
                        # t_p = dy/sqrt(g*(H-y[j])) # local confinement timescale (s)

                        # if np.random.rand() < p.free_fall_velocity*p.dt/p.dy:

                        # UP
                        if np.isnan(s[i, j + 1, k]):
                            P_u = 0
                        else:
                            P_u = 1.0 / p.swap_rate / s[i, j + 1, k]  # FIXME ????

                        # LEFT
                        if i > 0:
                            if (
                                np.isnan(s[i - 1, j + diag, k])
                                # or density(s, i - 1, j + diag) < p.critical_density
                                or density(s, i - 1, j + 1) <= A
                                # or density(s, i - 1, j + 1) <= A*p.critical_density
                            ):
                                P_l = 0  # P_r + P_l = 1 at s=1
                            else:
                                P_l = (0.5 + 0.5 * np.sin(np.radians(p.theta))) / s[i - 1, j + diag, k]

                            if hasattr(p, "internal_geometry"):
                                if p.boundary[i - 1, j + diag]:
                                    P_l *= p.internal_geometry["perf_rate"]
                            # if perf_plate and i-1==perf_pts[0]: P_l *= perf_rate
                            # if perf_plate and i-1==perf_pts[1]: P_l *= perf_rate
                        elif p.cyclic_BC:
                            if np.isnan(s[-1, j + diag, k]):
                                P_l = 0  # UP LEFT
                            else:
                                P_l = (0.5 + 0.5 * np.sin(np.radians(p.theta))) / s[-1, j + diag, k]
                        else:
                            P_l = 0

                        # RIGHT
                        if i + 1 < p.nx:
                            # if ( not np.isnan(s[i+1,j,k]) and not np.isnan(s[i+1,j+1,k]) ): # RIGHT
                            if (
                                np.isnan(s[i + 1, j + diag, k])
                                # or density(s, i + 1, j + diag) < p.critical_density
                                or density(s, i + 1, j + 1) <= A
                                # or density(s, i + 1, j + 1) <= A*p.critical_density
                            ):
                                P_r = 0
                            else:
                                P_r = (0.5 - 0.5 * np.sin(np.radians(p.theta))) / s[i + 1, j + diag, k]

                            if p.internal_geometry:
                                if p.boundary[i + 1, j + diag]:
                                    P_r *= p.internal_geometry["perf_rate"]
                            # if perf_plate and i+1==perf_pts[0]: P_r *= perf_rate
                            # if perf_plate and i+1==perf_pts[1]: P_r *= perf_rate
                        elif p.cyclic_BC:
                            if np.isnan(s[0, j + diag, k]):
                                P_r = 0
                            else:
                                P_r = (0.5 - 0.5 * np.sin(np.radians(p.theta))) / s[0, j + diag, k]
                        else:
                            P_r = 0

                        P_tot = P_u + P_l + P_r

                        if P_tot > 0:
                            P = np.random.rand()
                            if P < P_u / P_tot and P_u > 0:  # go up
                                dest = [i, j + 1, k]
                                v[i, j] += 1
                            elif P < (P_l + P_u) / P_tot:  # go left
                                dest = [i - 1, j + diag, k]
                                if diag == 0:
                                    u[i, j] += 1  # LEFT
                                    v[i, j] += 1
                                else:
                                    u[i, j] += np.sqrt(2)  # UP LEFT
                                    v[i, j] += np.sqrt(2)
                            else:  # go right
                                if i + 1 < p.nx:
                                    dest = [i + 1, j + diag, k]  # not a boundary
                                else:
                                    dest = [0, j + diag, k]  # right boundary
                                # if np.mean(~np.isnan(s[dest, j + 1, :])) > A:  # * (np.mean(np.isnan(s[i, j, :])) < 0.5):  # this sets the angle of repose
                                if diag == 0:
                                    u[i, j] -= 1  # RIGHT
                                    v[i, j] += 1
                                else:
                                    u[i, j] -= np.sqrt(2)  # UP RIGHT
                                    v[i, j] += np.sqrt(2)

                            # more than critical void threshold, we are a gas
                            # if np.mean(np.isnan(s[i, j, :])) > (1 - p.min_solid_density):
                            #     # print("aa")
                            #     s, c, T = swap([i, j, k], [i, j + 1, k], [s, c, T])
                            # else:
                            #     if (
                            #         np.mean(np.isnan(s[dest[0], j + 1, :])) < A
                            #     ):  # only swap in if there is support?
                            s, c, T = swap([i, j, k], dest, [s, c, T])
                            # u[i,j] += dest[0] - i
                            # v[i,j] += dest[1] - j

    return u, v, s, c, T


def swap(src, dest, arrays):
    for n in range(len(arrays)):
        if arrays[n] is not None:
            arrays[n][*src], arrays[n][*dest] = arrays[n][*dest], arrays[n][*src]
    return arrays


def add_voids(u, v, s, c, outlet):
    if p.add_voids == "central_outlet":  # Remove at central outlet - use this one
        for i in range(p.nx // 2 - p.half_width, p.nx // 2 + p.half_width + 1):
            for k in range(p.nm):
                # if np.random.rand() < Tg:
                if not np.isnan(s[i, 0, k]):
                    if p.refill:
                        if (
                            np.sum(
                                np.isnan(s[p.nx // 2 - p.half_width : p.nx // 2 + p.half_width + 1, -1, k])
                            )
                            > 0
                        ):
                            target = np.random.choice(np.nonzero(np.isnan(s[:, -1, k]))[0])
                            s[target, -1, k], s[i, 0, k] = s[i, 0, k], s[target, -1, k]
                    else:
                        s[i, 0, k] = np.nan
                    outlet[-1] += 1
    # elif temp_mode == "temperature":  # Remove at central outlet
    #     for i in range(nx // 2 - half_width, nx // 2 + half_width + 1):
    #         for k in range(nm):
    #             # if np.random.rand() < Tg:
    #             if not np.isnan(s[i, 0, k]):
    #                 if refill:
    #                     if np.sum(np.isnan(s[nx // 2 - half_width : nx // 2 + half_width + 1, -1, k])) > 0:
    #                         if internal_geometry:
    #                             target = (
    #                                 nx // 2
    #                                 - half_width
    #                                 + np.random.choice(
    #                                     np.nonzero(
    #                                         np.isnan(
    #                                             s[nx // 2 - half_width : nx // 2 + half_width + 1, -1, k]
    #                                         )
    #                                     )[0]
    #                                 )
    #                             )  # HACK
    #                         else:
    #                             target = np.random.choice(np.nonzero(np.isnan(s[:, -1, k]))[0])
    #                         s[target, -1, k], s[i, 0, k] = s[i, 0, k], s[target, -1, k]
    #                         T[target, -1, k] = inlet_temperature
    #                         outlet_T.append(T[i, 0, k])
    #                 else:
    #                     s[i, 0, k] = np.nan
    #                 outlet[-1] += 1
    elif p.add_voids == "multiple_outlets":  # Remove at multiple points in base
        for l, source_pt in enumerate(p.source_pts):
            for i in range(source_pt - p.half_width, source_pt + p.half_width + 1):
                for k in range(p.nm):
                    if np.random.rand() < p.Tg[l]:
                        target = np.random.choice(np.nonzero(np.isnan(s[:, -1, k]))[0])
                        s[target, -1, k] = s[i, 0, k]
                        if target <= p.internal_geometry.perf_pts[0]:
                            c[target, -1, k] = 0
                        elif target <= p.internal_geometry.perf_pts[1]:
                            c[target, -1, k] = 1
                        else:
                            c[target, -1, k] = 2
                        s[i, 0, k] = np.nan
    elif p.add_voids == "slope":  # Add voids at base
        for i in range(p.nx):
            for k in range(p.nm):
                if not np.isnan(s[i, 0, k]):
                    # MOVE UP TO FIRST VOID --- THIS GENERATES SHEARING WHEN INCLINED!
                    if (
                        np.random.rand() < (p.Tg * p.H) / (p.free_fall_velocity * p.dt)
                        and np.sum(np.isnan(s[i, :, k]))
                    ) > 0:  # Tg is relative height (out of the maximum depth) that voids should rise to before being filled
                        first_void = np.isnan(s[i, :, k]).nonzero()[0][0]
                        v[i, : first_void + 1] += np.isnan(s[i, : first_void + 1, k])
                        s[i, : first_void + 1, k] = np.roll(s[i, : first_void + 1, k], 1)
                    # MOVE EVERYTHING UP
                    # if (np.random.rand() < p.Tg * p.dt / p.dy and np.sum(np.isnan(s[i, :, k]))) > 0:
                    #     if np.isnan(s[i, -1, k]):
                    #         v[i, :] += 1  # np.isnan(s[i,:,k])
                    #         s[i, :, k] = np.roll(s[i, :, k], 1)
    elif p.add_voids == "mara":  # Add voids at base
        # for i in range(5,nx-5):
        for i in range(p.nx):
            for k in range(p.nm):
                if not np.isnan(s[i, 0, k]):
                    if np.random.rand() < p.Tg * p.dt / p.dy and np.sum(np.isnan(s[i, :, k])) > 0:
                        first_void = np.isnan(s[i, :, k]).nonzero()[0][0]
                        v[i, : first_void + 1] += np.isnan(s[i, : first_void + 1, k])
                        s[i, : first_void + 1, k] = np.roll(s[i, : first_void + 1, k], 1)
    # elif p.add_voids == "diff_test":
    #     if t == 0:
    #         s[nx // 2, 0, :] = np.nan
    elif p.add_voids == "pour":  # pour in centre at top
        s[p.nx // 2 - p.half_width : p.nx // 2 + p.half_width + 1, -1, :] = 1.0
    return u, v, s, c, outlet


# def generate_voids(u, v, s):  # Moving voids create voids
#     U = np.sqrt(u**2 + v**2)
#     for i in range(nx):
#         for j in range(ny):
#             for k in range(nm):
#                 if not np.isnan(s[i, j, k]):
#                     if np.random.rand() < 1 * U[i, j] / nm * dt / dy:  # FIXME
#                         last_void = (
#                             np.isfinite(s[i, :, k]).nonzero()[0][-1] + 1
#                         )  # get first void above top filled site
#                         # FIXME: THIS WILL DIE IF TOP HAS A VOID IN IT
#                         v[i, j : last_void + 1] += 1  # np.isnan(s[i,j:last_void+1,k])
#                         s[i, j : last_void + 1, k] = np.roll(s[i, j : last_void + 1, k], 1)
#     return u, v, s


def close_voids(u, v, s):
    """
    Not implemented. Do not use.
    """
    for i in range(p.nx):
        for j in np.arange(p.ny - 1, -1, -1):  # go from top to bottom
            for k in range(p.nm):
                if np.isnan(s[i, j, k]):
                    pass
                    # if np.random.rand() < 5e-2 * dt / dy:  # FIXME
                    #     v[i, j:] -= 1
                    #     s[i, j:, k] = np.roll(s[i, j:, k], -1)
    return u, v, s


def update_temperature(s, T, p):
    """
    Used for modelling the diffusion of heat into the body. Still not functional. Do not use.
    """
    T[np.isnan(s)] = p.temperature["inlet_temperature"]  # HACK
    T[p.boundary] = p.temperature["boundary_temperature"]
    T_inc = np.zeros_like(T)
    T_inc[1:-1, 1:-1] = 1e-3 * (T[2:, 1:-1] + T[:-2, 1:-1] + T[1:-1, 2:] + T[1:-1, :-2] - 4 * T[1:-1, 1:-1])
    return T + T_inc


def get_average(s):
    """
    Calculate the mean size over the microstructural co-ordinate.
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
    """
    depth = np.mean(np.mean(~np.isnan(s), axis=2), axis=1)
    return depth


def time_march(p):
    """
    Run the actual simulation(s) as defined in the input json file `p`.
    """

    if not os.path.exists(p.folderName):
        os.makedirs(p.folderName)
    if not hasattr(p, "internal_geometry"):
        p.internal_geometry = False
    if not hasattr(p, "cyclic_BC"):
        p.cyclic_BC = False
    if not hasattr(p, "theta"):
        p.theta = 0
    if not hasattr(p, "refill"):
        p.refill = False
    if not hasattr(p, "close_voids"):
        p.close_voids = False
    if not hasattr(p, "diag"):
        p.diag = 0
    # if not hasattr(p, "min_solid_density"):
    # p.min_solid_density = 0.5
    if not hasattr(p, "lagrangian"):
        p.lagrangian = False
    if not hasattr(p, "g"):
        p.g = 9.81
    if not hasattr(p, "critical_density"):
        p.critical_density = 0.5

    plotter.set_plot_size(p)

    y = np.linspace(0, p.H, p.ny)
    p.dy = y[1] - y[0]
    x = np.linspace(-p.nx * p.dy / 2, p.nx * p.dy / 2, p.nx)  # force equal grid spacing
    X, Y = np.meshgrid(x, y, indexing="ij")

    y += p.dy / 2.0
    t = 0

    p.t_p = p.s_m / np.sqrt(p.g * p.H)  # smallest confinement timescale (at bottom) (s)
    p.free_fall_velocity = np.sqrt(p.g * p.s_m)

    p.swap_rate = np.sqrt(4 * p.mu)

    # P_scaling = (p.mu**2) / 2.0
    # if P_scaling > 1:  # SOLVED: both swapping probabilities guaranteed to be less than or equal to 0.5
    #     p.P_adv = 0.5
    #     p.P_diff = p.P_adv / P_scaling
    # else:
    #     p.P_diff = 0.5
    #     p.P_adv = p.P_diff * P_scaling

    # p.dt = p.P_adv * p.dy / p.free_fall_velocity
    p.dt = 1 * p.dy / p.free_fall_velocity  # HACK: WHY ONE?!?
    # print(f"Time step is {p.dt} s")
    # p.dt = 1.0

    p.nt = int(np.ceil(p.t_f / p.dt))

    s_bar_time = np.zeros([p.nt, p.ny])
    nu_time = np.zeros_like(s_bar_time)
    nu_time_x = np.zeros([p.nt, p.nx])
    u_time = np.zeros_like(s_bar_time)

    s = IC(p)  # non-dimensional size
    u = np.zeros([p.nx, p.ny])
    v = np.zeros([p.nx, p.ny])

    if hasattr(p, "concentration"):
        c = np.zeros_like(s)  # original bin that particles started in
        c[int(p.internal_geometry.perf_pts[0] * p.nx) : int(p.internal_geometry.perf_pts[1] * p.nx)] = 1
        c[int(p.internal_geometry.perf_pts[1] * p.nx) :] = 2
        c[np.isnan(s)] = np.nan
    else:
        c = None

    if p.internal_geometry:
        p.boundary = np.zeros([p.nx, p.ny], dtype=bool)
        # boundary[4:-4:5,:] = 1
        p.boundary[np.cos(500 * 2 * np.pi * X) > 0] = 1
        p.boundary[:, : p.nx // 2] = 0
        p.boundary[:, -p.nx // 2 :] = 0
        p.boundary[:, p.ny // 2 - 5 : p.ny // 2 + 5] = 0
        p.boundary[np.abs(X) - 2 * p.half_width * p.dy > Y] = 1
        p.boundary[np.abs(X) - 2 * p.half_width * p.dy > p.H - Y] = 1
        boundary_tile = np.tile(p.boundary.T, [p.nm, 1, 1]).T
        s[boundary_tile] = np.nan
        # plt.pcolormesh(x,y,boundary.T)
        # plt.show()
        # sys.exit()
    else:
        p.boundary = np.zeros([p.nx, p.ny], dtype=bool)

    if hasattr(p, "temperature"):
        T = p.temperature["inlet_temperature"] * np.ones_like(s)
        T[np.isnan(s)] = np.nan
        outlet_T = []
    else:
        T = None

    plotter.plot_s(x, y, s, p, t)
    plotter.plot_nu(x, y, s, p, t)
    plotter.plot_u(x, y, s, u, v, p, t)
    if hasattr(p, "concentration"):
        plotter.plot_c(x, y, s, c, p, t)
    if hasattr(p, "temperature"):
        plotter.plot_T(x, y, s, T, p, t)
    outlet = []

    # print("Running " + p.folderName)
    # print('Tg = ' + str(Tg) + ', k_add = ' + str(p.free_fall_velocity*dt/(Tg*H)) + '\n')
    for t in tqdm(range(p.nt), leave=False, desc="Time"):
        outlet.append(0)
        u = np.zeros_like(u)
        v = np.zeros_like(v)

        # depth = get_depth(s)
        s_bar = get_average(s)
        # s_inv_bar = get_hyperbolic_average(s)
        if hasattr(p, "temperature"):
            T = update_temperature(s, T, p)  # delete the particles at the bottom of the hopper

        u, v, s, c, T = move_voids(u, v, s, p, c=c, T=T)

        # if t % 2 == 0:
        # u, v, s, c, T = move_voids_adv(u, v, s, c, T, boundary)
        # u, v, s, c, T = move_voids_diff(u, v, s, c, T, boundary)
        # else:
        # u, v, s, c, T = move_voids_diff(u, v, s, c, T, boundary)
        # u, v, s, c, T = move_voids_adv(u, v, s, c, T, boundary)

        u, v, s, c, outlet = add_voids(u, v, s, c, outlet)

        if p.close_voids:
            u, v, s = close_voids(u, v, s)

        if t % p.save_inc == 0:
            plotter.plot_s(x, y, s, p, t)
            plotter.plot_nu(x, y, s, p, t)
            plotter.plot_u(x, y, s, u, v, p, t)

            if hasattr(p, "concentration"):
                plotter.plot_c(x, y, s, c, p.folderName, t, p.internal_geometry)
            if hasattr(p, "save_outlet"):
                np.savetxt(p.folderName + "outlet.csv", np.array(outlet), delimiter=",")
            if hasattr(p, "temperature"):
                plotter.plot_T(x, y, s, T, p, t)
                np.savetxt(p.folderName + "outlet_T.csv", np.array(outlet_T), delimiter=",")
            if hasattr(p, "save_velocity"):
                np.savetxt(p.folderName + "u.csv", u / np.sum(np.isnan(s), axis=2), delimiter=",")
            if hasattr(p, "save_density_profile"):
                plotter.plot_profile(x, nu_time_x, p)

        s_bar_time[t] = s_bar  # save average size
        u_time[t] = np.mean(u, axis=0)
        nu_time[t] = np.mean(1 - np.mean(np.isnan(s), axis=2), axis=0)
        nu_time_x[t] = np.mean(1 - np.mean(np.isnan(s), axis=2), axis=1)
        t += 1
        # if t % 10 == 0:
        # print(" t = " + str(t * dt) + "                ", end="\r")
    plotter.plot_s(x, y, s, p, t)
    plotter.plot_nu(x, y, s, p, t)
    plotter.plot_u(x, y, s, u, v, p, t)
    plotter.plot_s_bar(y, s_bar_time, nu_time, p)
    plotter.plot_u_time(y, u_time, nu_time, p)
    np.save(p.folderName + "nu_t_x.npy", nu_time_x)
    if hasattr(p, "concentration"):
        plotter.plot_c(c)
    if hasattr(p, "save_outlet"):
        np.savetxt(p.folderName + "outlet.csv", np.array(outlet), delimiter=",")
    if hasattr(p, "save_velocity"):
        np.savetxt(p.folderName + "u.csv", u / np.sum(np.isnan(s), axis=2), delimiter=",")
    if hasattr(p, "save_density_profile"):
        plotter.plot_profile(x, nu_time_x, p)


if __name__ == "__main__":
    with open(sys.argv[1], "r") as params:
        # parse file
        dict = json5.loads(params.read())
        dict["input_filename"] = (sys.argv[1].split("/")[-1]).split(".")[0]
        p_init = dict_to_class(dict)

        all_sims = list(product(*p_init.lists))

        for sim in tqdm(all_sims, desc="Sim", leave=False):
            folderName = f"output/{dict['input_filename']}/"
            dict_copy = dict.copy()
            for i, key in enumerate(p_init.list_keys):
                dict_copy[key] = sim[i]
                folderName += f"{key}_{sim[i]}/"
            p = dict_to_class(dict_copy)
            p.folderName = folderName
            time_march(p)

        # for i in tqdm(p_init.mu, desc="Friction angle", disable=(len(p_init.mu) == 1)):
        #     p.foldername += f"mu_{i}/"
        #     for j in tqdm(
        #         p_init.half_width, desc="Half width", leave=False, disable=(len(p_init.half_width) == 1)
        #     ):
        #         p = dict_to_class(dict)
        #         p.mu = i
        #         p.half_width = j
        #         p.folderName = f"output/{p.input_filename}/mu_{i}/half_width_{j}/"
        #         time_march(p)
