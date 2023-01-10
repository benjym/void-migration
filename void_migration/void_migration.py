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

# warnings.filterwarnings("ignore")


class dict_to_class(dict):
    """
    A convenience class to store the information from the parameters dictionary. Used because I prefer using p.variable to p['variable'].
    """

    def __init__(self, dict):
        lists = []
        for key in dict:
            setattr(self, key, dict[key])
            if isinstance(dict[key], list):
                lists.append(key)
        setattr(self, "lists", lists)


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
        p.s_m = 1
    if p.gsd_mode == "bi":  # bidisperse
        s = np.random.choice([p.s_m, 1], size=[p.nx, p.ny, p.nm])
    elif p.gsd_mode == "poly":  # polydisperse
        s_0 = p.s_m / (1.0 - p.s_m)  # intermediate calculation
        s = np.random.rand(p.nx, p.ny, p.nm)
        s = (s + s_0) / (s_0 + 1.0)  # now between s_m and 1

    # where particles are in space
    if p.IC_mode == "random":  # voids everywhere randomly
        mask = np.random.rand(p.nx, p.ny, p.nm) < p.fill_ratio
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


def move_voids_adv(u, v, s, c, T, boundary):  # Advection
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


def move_voids_diff(u, v, s, c, T, boundary):  # Diffusion
    """
    Deprecated function. Do not use.
    """
    for j in range(p.ny - 2, -1, -1):
        x_loop = np.arange(p.nx)
        if p.internal_geometry:
            x_loop = x_loop[~boundary[:, j]]  # don't move apparent voids at boundaries
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
                            if boundary[i - 1, j]:
                                l = 1e10  # zero chance of moving there
                            else:
                                l = s[i - 1, j, k]
                            if boundary[i + 1, j]:
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


def move_voids(u, v, s, diag=0, c=None, T=None, boundary=None):  # pick between left, up or right
    """
    Function to move voids each timestep.

    Args:
        u: Storage container for counting how many voids moved horizontally
        v: Storage container for counting how many voids moved vertically
        s: 3D array containing the local sizes everywhere. `NaN`s represent voids. Other values represent the grain size. The first two dimensions represent real space, the third dimension represents the micro-structural coordinate.
        diag: Should the voids swap horizontally (von neumnann neighbourhood, `diag=0) or diagonally upwards (moore neighbourhood, `diag=1`). Default value `0`.
        c: If array_like, a storage container for tracking motion of differently labelled particles. If `None`, do nothing.
        T: If array_like, the temperature field. If `None`, do nothing.
        boundary: If array_like, a descriptor of cells which voids cannot move into (i.e. boundaries). If `internal_boundary` is defined in the params file, allow for reduced movement rates rather than zero. If `None`, do nothing.

    Returns:
        u: The updated horizontal velocity
        v: The updated vertical velocity
        s: The new locations of the grains
        c: The updated concentration field
        T: The updated temperature field
    """

    # WHERE THE HELL DID THIS COME FROM???
    if p.mu < 1:
        A = p.mu / 2.0  # proportion of cell that should be filled diagonally up
    else:
        A = 1.0 - 1.0 / (2.0 * p.mu)
    # print(A)
    # A = 0.01

    for j in range(p.ny - 2, -1, -1):
        if p.internal_geometry:
            x_loop = np.arange(p.nx)[~boundary[:, j]]  # don't move apparent voids at boundaries
        else:
            x_loop = np.arange(p.nx)
        np.random.shuffle(x_loop)
        for i in x_loop:
            for k in range(p.nm):
                if np.isnan(s[i, j, k]):
                    # t_p = dy/sqrt(g*(H-y[j])) # local confinement timescale (s)

                    # if np.random.rand() < free_fall_velocity*dt/dy:

                    # UP
                    if np.isnan(s[i, j + 1, k]):
                        P_u = 0
                    else:
                        P_u = 1.0 / p.swap_rate / s[i, j + 1, k]  # FIXME ????

                    # LEFT
                    if i > 0:
                        if np.isnan(s[i - 1, j + diag, k]) or np.mean(np.isnan(s[i - 1, j + 1, :])) > A:
                            P_l = 0  # P_r + P_l = 1 at s=1
                        else:
                            P_l = (0.5 + 0.5 * np.sin(np.radians(p.theta))) / s[i - 1, j + diag, k]

                        if hasattr(p, "internal_geometry"):
                            if boundary[i - 1, j + diag]:
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
                        if np.isnan(s[i + 1, j + diag, k]) or np.mean(np.isnan(s[i + 1, j + 1, :])) > A:
                            P_r = 0
                        else:
                            P_r = (0.5 - 0.5 * np.sin(np.radians(p.theta))) / s[i + 1, j + diag, k]

                        if p.internal_geometry:
                            if boundary[i + 1, j + diag]:
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
                            # s[i, j, k], s[i, j + 1, k] = s[i, j + 1, k], s[i, j, k]
                            # if c is not None:
                            #     c[i, j, k], c[i, j + 1, k] = c[i, j + 1, k], c[i, j, k]
                            # if T is not None:
                            #     T[i, j, k], T[i, j + 1, k] = T[i, j + 1, k], T[i, j, k]
                            v[i, j] += 1
                        elif P < (P_l + P_u) / P_tot:  # go left
                            # print(np.mean(~np.isnan(s[i - 1, j + 1, :])) > A)
                            # if np.mean(~np.isnan(s[i - 1, j + 1, :])) > A:  # * (np.mean(np.isnan(s[i, j, :])) < 0.5):  # if here is mostly empty (ie outside mass) --- this sets the angle of repose
                            dest = [i - 1, j + diag, k]
                            # s[i, j, k], s[i - 1, j + diag, k] = s[i - 1, j + diag, k], s[i, j, k]
                            # if c is not None:
                            #     c[i, j, k], c[i - 1, j + diag, k] = c[i - 1, j + diag, k], c[i, j, k]
                            # if T is not None:
                            #     T[i, j, k], T[i - 1, j + diag, k] = T[i - 1, j + diag, k], T[i, j, k]
                            if diag == 0:
                                u[i, j] += 1  # LEFT
                                v[i, j] += 1
                            else:
                                u[i, j] += np.sqrt(2)  # UP LEFT
                                v[i, j] += np.sqrt(2)
                        elif P < (P_l + P_u + P_r) / P_tot:  # go right
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

                        s[i, j, k], s[*dest] = s[*dest], s[i, j, k]
                        if c is not None:
                            c[i, j, k], c[*dest] = c[*dest], c[i, j, k]
                        if T is not None:
                            T[i, j, k], T[*dest] = T[*dest], T[i, j, k]
    return u, v, s, c, T


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
                    # MOVE UP TO FIRST VOID
                    # if ( np.random.rand() < (free_fall_velocity*dt)/(Tg*H) and sum(isnan(s[i,:,k])) ) > 0: # Tg is relative height (out of the maximum depth) that voids should rise to before being filled
                    # first_void = np.isnan(s[i,:,k]).nonzero()[0][0]
                    # v[i,:first_void+1] += np.isnan(s[i,:first_void+1,k])
                    # s[i,:first_void+1,k] = roll(s[i,:first_void+1,k],1)
                    # MOVE EVERYTHING UP
                    if (np.random.rand() < p.Tg * p.dt / p.dy and np.sum(np.isnan(s[i, :, k]))) > 0:
                        if np.isnan(s[i, -1, k]):
                            v[i, :] += 1  # np.isnan(s[i,:,k])
                            s[i, :, k] = np.roll(s[i, :, k], 1)
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


def update_temperature(s, T, boundary):
    """
    Used for modelling the diffusion of heat into the body. Still not functional. Do not use.
    """
    T[np.isnan(s)] = p.temperature["inlet_temperature"]  # HACK
    T[boundary] = p.temperature["boundary_temperature"]
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
    # nx = 20
    # ny = 20  # 200
    # nm = 20
    # t_f = 2  # final time (s)
    # CFL = 0.2  # stability criteria, < 0.5
    # theta = 0.0
    # H = 0.1  # m
    # # s_M = 1e-3  # maximum particle size (m)
    # g = 9.81  # m/s^2
    # mode = "bi"
    # # mode = 'poly'
    # # close = True
    # close = False
    # if mode == "bi":
    #     s_m = 0.1
    # elif mode == "poly":
    #     s_m = 0.2  # minimum size

    # temp_mode = sys.argv[1]
    # perf_plate = False
    # perf_pts = None
    # refill = False
    # internal_geometry = False

    # elif temp_mode == "WGCP":
    #     cyclic_BC = False
    #     perf_plate = True
    #     perf_rate = 0.01  # drop in probability across perf plate
    #     source_pts = [nx // 6, nx // 2, 5 * nx // 6]
    #     perf_pts = [nx // 3, 2 * nx // 3]
    #     # u_applied = zeros_like(y)
    #     Tg_l = 0.01
    #     Tg_m = Tg_l / 2.0
    #     Tg_r = Tg_l / 10.0
    #     Tg = [Tg_l, Tg_m, Tg_r]
    #     gamma_dot = 0.0
    #     half_width = 3
    #     IC_mode = "top"
    #     top_fill = 0.7
    #     save_inc = 10
    #     mu = float(sys.argv[2])
    # elif temp_mode == "slope":
    #     cyclic_BC = True
    #     mu = float(sys.argv[2])
    #     theta = float(sys.argv[3])
    #     Tg = float(sys.argv[4])
    #     ny = int(sys.argv[5])
    #     nm = 20
    #     nx = 5
    #     IC_mode = "random"
    #     fill = 0.2
    #     save_inc = 10000
    #     t_f = 10.0  # s
    # elif temp_mode == "mara":
    #     cyclic_BC = True
    #     mu = float(sys.argv[2])
    #     theta = 0.0
    #     Tg = float(sys.argv[3])
    #     IC_mode = "top"
    #     top_fill = 0.5
    #     ny = 50
    #     nx = ny * 4
    #     nm = 5
    #     save_inc = 10
    #     t_f = 20.0  # s
    # elif temp_mode == "pour":
    #     cyclic_BC = False
    #     mu = float(sys.argv[2])
    #     half_width = int(sys.argv[3])
    #     theta = 0.0
    #     # Tg = float(sys.argv[4])
    #     fill = 0.0
    #     ny = 100
    #     nx = 50
    #     nm = 1
    #     IC_mode = "empty"
    #     save_inc = 1000
    #     t_f = 20.0  # s

    # if temp_mode == "WGCP":
    #     folderName = "plots/" + temp_mode + "/" + mode + "/" + str(Tg_l) + "/"
    # elif temp_mode == "hopper":
    #     folderName = "plots/" + temp_mode + "/mu_" + str(mu) + "/D_" + str(half_width) + "/"
    # elif temp_mode == "slope":
    #     folderName = (
    #         "plots/"
    #         + temp_mode
    #         + "/"
    #         + mode
    #         + "/mu_"
    #         + str(mu)
    #         + "/theta_"
    #         + str(theta)
    #         + "/Tg_"
    #         + str(Tg)
    #         + "/ny_"
    #         + str(ny)
    #         + "/"
    #     )
    # elif temp_mode == "mara":
    #     folderName = "plots/" + temp_mode + "/mu_" + str(mu) + "/Tg_" + str(Tg) + "/"
    # else:
    #     folderName = "plots/" + temp_mode + "/mu_" + str(mu) + "/"
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

    # fig = plt.figure(figsize=[nx / 10., ny / 10.])
    y = np.linspace(0, p.H, p.ny)
    p.dy = y[1] - y[0]
    x = np.linspace(-p.nx * p.dy / 2, p.nx * p.dy / 2, p.nx)  # force equal grid spacing
    X, Y = np.meshgrid(x, y, indexing="ij")

    y += p.dy / 2.0
    t = 0

    # t_p = s_m/sqrt(g*H) # smallest confinement timescale (at bottom) (s)
    # free_fall_velocity = np.sqrt(g * p.s_m)

    p.swap_rate = np.sqrt(4 * p.mu)

    P_scaling = (p.mu**2) / 2.0
    if P_scaling > 1:  # SOLVED: both swapping probabilities guaranteed to be less than or equal to 0.5
        p.P_adv = 0.5
        p.P_diff = p.P_adv / P_scaling
    else:
        p.P_diff = 0.5
        p.P_adv = p.P_diff * P_scaling

    # p.dt = p.P_adv * p.dy / free_fall_velocity
    # print(p.dt)
    p.dt = 1.0

    nt = int(np.ceil(p.t_f / p.dt))

    s_bar_time = np.zeros([nt, p.ny])
    nu_time = np.zeros_like(s_bar_time)
    nu_time_x = np.zeros([nt, p.nx])
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
        boundary = np.zeros([p.nx, p.ny], dtype=bool)
        # boundary[4:-4:5,:] = 1
        boundary[np.cos(500 * 2 * np.pi * X) > 0] = 1
        boundary[:, : p.nx // 2] = 0
        boundary[:, -p.nx // 2 :] = 0
        boundary[:, p.ny // 2 - 5 : p.ny // 2 + 5] = 0
        boundary[np.abs(X) - 2 * p.half_width * p.dy > Y] = 1
        boundary[np.abs(X) - 2 * p.half_width * p.dy > p.H - Y] = 1
        boundary_tile = np.tile(boundary.T, [p.nm, 1, 1]).T
        s[boundary_tile] = np.nan
        # plt.pcolormesh(x,y,boundary.T)
        # plt.show()
        # sys.exit()
    else:
        boundary = np.zeros([p.nx, p.ny], dtype=bool)

    if hasattr(p, "temperature"):
        T = p.temperature["inlet_temperature"] * np.ones_like(s)
        T[np.isnan(s)] = np.nan
        outlet_T = []
    else:
        T = None

    plotter.plot_s(x, y, s, p.folderName, t, p.internal_geometry, p.s_m)
    plotter.plot_nu(x, y, s, p.folderName, t, p.internal_geometry, boundary)
    plotter.plot_u(x, y, s, u, v, p.folderName, t, p.nm, p.IC_mode, p.internal_geometry, boundary)
    if hasattr(p, "concentration"):
        plotter.plot_c(x, y, s, c, p.folderName, t, p.internal_geometry)
    if hasattr(p, "temperature"):
        plotter.plot_T(
            x,
            y,
            s,
            T,
            p.folderName,
            t,
            p.temperature["boundary_temperature"],
            p.temperature["inlet_temperature"],
        )
    outlet = []

    # print("Running " + p.folderName)
    # print('Tg = ' + str(Tg) + ', k_add = ' + str(free_fall_velocity*dt/(Tg*H)) + '\n')
    for t in tqdm(range(nt), leave=False, desc="Time"):
        outlet.append(0)
        u = np.zeros_like(u)
        v = np.zeros_like(v)

        # depth = get_depth(s)
        s_bar = get_average(s)
        # s_inv_bar = get_hyperbolic_average(s)
        if hasattr(p, "temperature"):
            T = update_temperature(s, T, boundary)  # delete the particles at the bottom of the hopper

        u, v, s, c, T = move_voids(u, v, s, p.diag, c, T, boundary)

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
            plotter.plot_s(x, y, s, p.folderName, t, p.internal_geometry, p.s_m)
            plotter.plot_nu(x, y, s, p.folderName, t, p.internal_geometry, boundary)
            plotter.plot_u(x, y, s, u, v, p.folderName, t, p.nm, p.IC_mode, p.internal_geometry, boundary)

            if hasattr(p, "concentration"):
                plotter.plot_c(x, y, s, c, p.folderName, t, p.internal_geometry)
            if hasattr(p, "save_outlet"):
                np.savetxt(p.folderName + "outlet.csv", np.array(outlet), delimiter=",")
            if hasattr(p, "temperature"):
                plotter.plot_T(
                    x,
                    y,
                    s,
                    T,
                    p.folderName,
                    t,
                    p.temperature["boundary_temperature"],
                    p.temperature["inlet_temperature"],
                )
                np.savetxt(p.folderName + "outlet_T.csv", np.array(outlet_T), delimiter=",")
            if hasattr(p, "save_velocity"):
                np.savetxt(p.folderName + "u.csv", u / np.sum(np.isnan(s), axis=2), delimiter=",")
            if hasattr(p, "save_density_profile"):
                plotter.plot_profile(x, nu_time_x, p.folderName, nt, p.t_f)

        s_bar_time[t] = s_bar  # save average size
        u_time[t] = np.mean(u, axis=0)
        nu_time[t] = np.mean(1 - np.mean(np.isnan(s), axis=2), axis=0)
        nu_time_x[t] = np.mean(1 - np.mean(np.isnan(s), axis=2), axis=1)
        t += 1
        # if t % 10 == 0:
        # print(" t = " + str(t * dt) + "                ", end="\r")
    plotter.plot_s(x, y, s, p.folderName, t, p.internal_geometry, p.s_m)
    plotter.plot_nu(x, y, s, p.folderName, t, p.internal_geometry, boundary)
    plotter.plot_u(x, y, s, u, v, p.folderName, t, p.nm, p.IC_mode, p.internal_geometry, boundary)
    plotter.plot_s_bar(s_bar_time, nu_time, p.s_m, p.folderName)
    plotter.plot_u_time(y, u_time, nu_time, p.folderName, nt)
    np.save(p.folderName + "nu_t_x.npy", nu_time_x)
    if hasattr(p, "concentration"):
        plotter.plot_c(c)
    if hasattr(p, "save_outlet"):
        np.savetxt(p.folderName + "outlet.csv", np.array(outlet), delimiter=",")
    if hasattr(p, "save_velocity"):
        np.savetxt(p.folderName + "u.csv", u / np.sum(np.isnan(s), axis=2), delimiter=",")
    if hasattr(p, "save_density_profile"):
        plotter.plot_profile(x, nu_time_x, p.folderName, nt, p.t_f)


if __name__ == "__main__":
    with open(sys.argv[1], "r") as params:
        # parse file
        dict = json5.loads(params.read())
        dict["input_filename"] = (sys.argv[1].split("/")[-1]).split(".")[0]
        p_init = dict_to_class(dict)

        for i in tqdm(p_init.mu, desc="Friction angle", disable=(len(p_init.mu) == 1)):
            for j in tqdm(
                p_init.half_width, desc="Half width", leave=False, disable=(len(p_init.half_width) == 1)
            ):
                p = dict_to_class(dict)
                p.mu = i
                p.half_width = j
                p.folderName = f"output/{p.input_filename}/mu_{i}/half_width_{j}/"
                time_march(p)
