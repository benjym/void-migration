#!/usr/bin/python

__doc__ = """
void_migration.py

This script simulates ....
"""
__author__ = "Benjy Marks"
__version__ = "0.3"

import sys
import numpy as np
import concurrent.futures
from tqdm import tqdm
from itertools import product
from numpy.typing import ArrayLike

# from scipy.ndimage import maximum_filter

# from numba import jit, njit
import plotter
import params
import operators
import thermal


def IC(p):
    """
    Sets up the initial value of the grain size and/or void distribution everywhere.

    Args:
        p: Parameters class. In particular, the `gsd_mode` and `IC_mode` should be set to determine the grain size distribution (gsd) and the initial condition (IC).

    Returns:
        The array of grain sizes. Values of `NaN` are voids.
    """
    rng = np.random.default_rng()
    pre_masked = False

    # pick a grain size distribution
    if p.gsd_mode == "mono":
        s = np.nan * np.ones([p.nx, p.ny, p.nm])  # monodisperse
        for i in range(p.nx):
            for j in range(p.ny):
                fill = rng.choice(p.nm, size=int(p.nm * p.nu_fill), replace=False)
                s[i, j, fill] = p.s_m
        p.s_M = p.s_m
    if p.gsd_mode == "bi":  # bidisperse
        if (p.nm * p.large_concentration * p.nu_fill) < 2:
            s = np.random.choice([p.s_m, p.s_M], size=[p.nx, p.ny, p.nm])
        else:
            s = np.nan * np.ones([p.nx, p.ny, p.nm])
            for i in range(p.nx):
                for j in range(p.ny):
                    large = rng.choice(
                        p.nm, size=int(p.nm * p.large_concentration * p.nu_fill), replace=False
                    )
                    s[i, j, large] = p.s_M
                    remaining = np.where(np.isnan(s[i, j, :]))[0]
                    small = rng.choice(
                        remaining, size=int(p.nm * (1 - p.large_concentration) * p.nu_fill), replace=False
                    )
                    s[i, j, small] = p.s_m
            pre_masked = True
    elif p.gsd_mode == "poly":  # polydisperse
        # s_0 = p.s_m / (1.0 - p.s_m)  # intermediate calculation
        s_non_dim = np.random.rand(p.nm)
        # s = (s + s_0) / (s_0 + 1.0)  # now between s_m and 1
        this_s = (p.s_M - p.s_m) * s_non_dim + p.s_m
        s = np.nan * np.ones([p.nx, p.ny, p.nm])
        # HACK: gsd least uniform in space, still need to account for voids
        for i in range(p.nx):
            for j in range(p.ny):
                np.shuffle(this_s)
                s[i, j, :] = this_s

    # where particles are in space
    if not pre_masked:
        if p.IC_mode == "random":  # voids everywhere randomly
            mask = np.random.rand(p.nx, p.ny, p.nm) < p.nu_fill
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

            mask[
                :, -1, :
            ] = True  # top row can't be filled for algorithmic reasons - could solve this if we need to
        elif p.IC_mode == "empty":  # completely empty
            mask = np.ones([p.nx, p.ny, p.nm], dtype=bool)

        s[mask] = np.nan

    return s


def stable_slope(
    s: ArrayLike,
    i: int,
    j: int,
    dest: int,
    p: params.dict_to_class,
    nu: ArrayLike,
    dnu_dx: ArrayLike,
    dnu_dy: ArrayLike,
) -> bool:
    """Determine whether a void should swap with a solid particle.

    Args:
        i: an integer representing a row index
        j: an integer representing a column index
        lr: the cell we want to move into

    Returns:
        True if the void should NOT swap with a solid particle (i.e. the slope is stable). False otherwise.
    """
    if ((nu[dest, j] > nu[i, j]) and (nu[dest, j] - nu[i, j]) < p.mu * p.nu_cs) and (nu[i, j + 1] == 0):
        return True
    else:
        return False


# @njit
def move_voids(
    u: ArrayLike,
    v: ArrayLike,
    s: ArrayLike,
    p: params.dict_to_class,
    diag: int = 0,
    c: None | ArrayLike = None,
    T: None | ArrayLike = None,
    N_swap: None | ArrayLike = None,
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

    # if p.swap_rate == "constant":
    #     swap_rate = np.ones_like(s[:, :, 0])
    # if N_swap is None:
    #     swap_rate = np.ones_like(s[:, :, 0])
    # else:
    #     e = 0.8
    #     s_bar = get_hyperbolic_average(s)
    #     nu = 1.0 - np.mean(np.isnan(s[:, :, :]), axis=2)
    #     nu_RCP = 0.64
    #     nu_F = p.nu_cs
    #     Tg = N_swap/p.nm*p.dy*p.dy/(p.dt*p.dt)
    #     D_KT = np.sqrt(np.pi)/(8*(1+e))*s_bar/(nu*5.6916*(nu_RCP-nu_F)/(nu_RCP-nu))*np.sqrt(Tg)
    #     swap_rate = D_KT*2*p.dt/(p.dy*p.dy)
    #     # print(np.nanmin(swap_rate),np.nanmax(swap_rate))
    #     # import matplotlib.pyplot as plt
    #     # plt.figure(99)
    #     # plt.clf()
    #     # plt.ion()
    #     # plt.imshow(N_swap)
    #     # plt.colorbar()
    #     # plt.pause(1)

    # N_swap = np.ones_like(s[:, :, 0]) # HACK - SET NON-ZERO Tg EVERYWHERE FOR TESTING

    # y_loop = np.arange(p.ny - 2, -1, -1)
    # np.random.shuffle(y_loop)
    # for j in y_loop:
    #     if p.internal_geometry:
    #         x_loop = np.arange(p.nx)[~p.boundary[:, j]]  # don't move apparent voids at boundaries
    #     else:
    #         x_loop = np.arange(p.nx)
    #     np.random.shuffle(x_loop)
    #     for i in x_loop:
    #         m_loop = np.arange(p.nm)
    #         np.random.shuffle(m_loop)
    #         for k in m_loop:
    nu = 1.0 - np.mean(np.isnan(s[:, :, :]), axis=2)
    # kernel = np.array([[0,1,0],[1,0,1],[0,0,0]]).T

    # nu_max = maximum_filter(nu, footprint=kernel)#, mode='constant', cval=0.0)
    # import matplotlib.pyplot as plt
    # plt.figure(7)
    # plt.subplot(211)
    # plt.imshow(nu.T)
    # plt.colorbar()
    # plt.subplot(212)
    # plt.imshow(nu_max.T)
    # plt.colorbar()
    # plt.pause(0.001)

    dnu_dx, dnu_dy = np.gradient(nu)
    s_bar = operators.get_average(s)
    s_inv_bar = operators.get_hyperbolic_average(
        s
    )  # HACK: SHOULD RECALCULATE AFTER EVERY SWAP — WILL BE SUPER SLOW??
    # s_inv_bar[np.isnan(s_inv_bar)] = 1.0 / (1.0 / p.s_m + 1.0 / p.s_M)  # FIXME
    # s_bar[np.isnan(s_bar)] = (p.s_m + p.s_M)/2.  # FIXME

    for index in params.indices:
        i, j, k = np.unravel_index(index, [p.nx, p.ny - 1, p.nm])
        if (
            nu[i, j] < p.nu_cs
        ):  # and nu_max[i,j] > 0:  # the material is a liquid (below critical solid fraction) AND there is some material nearby
            # if ((nu[i, j] < p.nu_cs) and (np.abs(dnu_dx[i,j]) > p.mu*p.nu_cs/2.)):  # the material is a liquid (below critical solid fraction AND slope is high)
            if np.isnan(s[i, j, k]):
                # print(s_inv_bar[i,j])

                # t_p = dy/sqrt(g*(H-y[j])) # local confinement timescale (s)

                # if np.random.rand() < p.free_fall_velocity*p.dt/p.dy:

                # UP
                if np.isnan(s[i, j + 1, k]):
                    P_u = 0
                else:
                    P_u = p.P_u_ref * (s_inv_bar[i, j + 1] / s[i, j + 1, k])

                # LEFT
                if i == 0:
                    if p.cyclic_BC:
                        l = -1
                    else:
                        l = i  # will force P_l to be zero at boundary
                else:
                    l = i - 1

                # if np.isnan(s[l, j + diag, k]):
                if np.isnan(s[l, j + diag, k]) or stable_slope(s, i, j, l, p, nu, dnu_dx, dnu_dy):
                    P_l = 0  # P_r + P_l = 1 at s=1
                else:
                    # P_l = (0.5 + 0.5 * np.sin(np.radians(p.theta))) / (s[l, j + diag, k]/s_inv_bar[i,j])
                    # P_l = p.P_lr_ref * (s_inv_bar[i, j] / s[l, j + diag, k])
                    P_l = p.P_lr_ref * (s[l, j + diag, k] / s_bar[l, j + diag])

                # if hasattr(p, "internal_geometry"):
                #     if p.boundary[l, j + diag]:
                #         P_l *= p.internal_geometry["perf_rate"]
                # if perf_plate and i-1==perf_pts[0]: P_l *= perf_rate
                # if perf_plate and i-1==perf_pts[1]: P_l *= perf_rate

                # RIGHT
                if i == p.nx - 1:
                    if p.cyclic_BC:
                        r = 0
                    else:
                        r = i  # will force P_r to be zero at boundary
                else:
                    r = i + 1

                # if np.isnan(s[r, j + diag, k]):
                if np.isnan(s[r, j + diag, k]) or stable_slope(s, i, j, r, p, nu, dnu_dx, dnu_dy):
                    P_r = 0
                else:
                    # P_r = (0.5 - 0.5 * np.sin(np.radians(p.theta))) / (s[r, j + diag, k]/s_inv_bar[i,j])
                    # P_r = p.P_lr_ref * (s_inv_bar[i, j] / s[r, j + diag, k])
                    P_r = p.P_lr_ref * (s[r, j + diag, k] / s_bar[r, j + diag])

                # if p.internal_geometry:
                #     if p.boundary[r, j + diag]:
                #         P_r *= p.internal_geometry["perf_rate"]
                # if perf_plate and i+1==perf_pts[0]: P_r *= perf_rate
                # if perf_plate and i+1==perf_pts[1]: P_r *= perf_rate

                P_tot = P_u + P_l + P_r
                # print(P_tot)
                if P_tot > 1:
                    print(f"Error: P_tot > 1, P_u = {P_u}, P_l = {P_l}, P_r = {P_r}")

                dest = None
                if P_tot > 0:
                    P = np.random.rand()
                    if P < P_u and P_u > 0:  # go up
                        dest = [i, j + 1, k]
                        # v[i, j] += 1
                        if not np.isnan(s[i, j + 1, k]):
                            v[i, j] += 1
                    elif P < (P_l + P_u):  # go left
                        dest = [l, j + diag, k]

                        if diag == 0:
                            u[i, j] += 1  # LEFT
                            v[i, j] += 1
                        else:
                            u[i, j] += np.sqrt(2)  # UP LEFT
                            v[i, j] += np.sqrt(2)
                    elif P < P_tot:  # go right
                        dest = [r, j + diag, k]

                        if diag == 0:
                            u[i, j] -= 1  # RIGHT
                            v[i, j] += 1
                        else:
                            u[i, j] -= np.sqrt(2)  # UP RIGHT
                            v[i, j] += np.sqrt(2)
                    else:
                        pass
                        # print('NOTHING')

                    if dest is not None:
                        [s, c, T], nu = operators.swap([i, j, k], dest, [s, c, T], nu, p)

                    # N_swap[i, j] += 1
                    # N_swap[dest[0],dest[1]] += 1

    return u, v, s, c, T, N_swap


def add_voids(u, v, s, p, c, outlet):
    if p.add_voids == "central_outlet":  # Remove at central outlet - use this one
        for i in range(p.nx // 2 - p.half_width, p.nx // 2 + p.half_width + 1):
            for k in range(p.nm):
                # if np.random.rand() < 0.1:
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


def time_march(p):
    """
    Run the actual simulation(s) as defined in the input json file `p`.
    """

    plotter.set_plot_size(p)

    y = np.linspace(0, p.H, p.ny)
    p.dy = y[1] - y[0]
    x = np.linspace(-p.nx * p.dy / 2, p.nx * p.dy / 2, p.nx)  # force equal grid spacing
    X, Y = np.meshgrid(x, y, indexing="ij")

    y += p.dy / 2.0
    t = 0

    p.t_p = p.s_m / np.sqrt(p.g * p.H)  # smallest confinement timescale (at bottom) (s)
    p.free_fall_velocity = np.sqrt(p.g * (p.s_m + p.s_M) / 2.0)  # time to fall one mean diameter (s)

    safe = False
    stability = 0.5
    while not safe:
        p.P_u_ref = stability
        p.dt = p.P_u_ref * p.dy / p.free_fall_velocity
        p.P_lr_ref = p.alpha * p.P_u_ref

        p.P_u_max = p.P_u_ref * (p.s_M / p.s_m)
        p.P_lr_max = p.P_lr_ref * (p.s_M / p.s_m)

        if p.P_u_max + 2 * p.P_lr_max <= 1:
            safe = True
        else:
            stability *= 0.95

    p.nt = int(np.ceil(p.t_f / p.dt))

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
    else:
        p.boundary = np.zeros([p.nx, p.ny], dtype=bool)

    if hasattr(p, "temperature"):
        T = p.temperature["inlet_temperature"] * np.ones_like(s)
        T[np.isnan(s)] = np.nan
    else:
        T = None

    plotter.save_coordinate_system(x, y, p)
    plotter.update(x, y, s, u, v, c, T, p, t)
    outlet = []
    N_swap = None

    params.indices = np.arange(p.nx * (p.ny - 1) * p.nm)
    np.random.shuffle(params.indices)

    for t in tqdm(range(1, p.nt), leave=False, desc="Time", position=p.concurrent_index + 1):
        outlet.append(0)
        u = np.zeros_like(u)
        v = np.zeros_like(v)

        if hasattr(p, "temperature"):
            T = thermal.update_temperature(s, T, p)

        u, v, s, c, T, N_swap = move_voids(u, v, s, p, c=c, T=T, N_swap=N_swap)

        u, v, s, c, outlet = add_voids(u, v, s, p, c, outlet)

        if p.close_voids:
            u, v, s = close_voids(u, v, s)

        if t % p.save_inc == 0:
            plotter.update(x, y, s, u, v, c, T, p, t)

        t += 1

    plotter.update(x, y, s, u, v, c, T, outlet, p, t)


def run_simulation(sim_with_index):
    index, sim = sim_with_index
    with open(sys.argv[1], "r") as f:
        dict, p_init = params.load_file(f)
    folderName = f"output/{dict['input_filename']}/"
    dict_copy = dict.copy()
    for i, key in enumerate(p_init.list_keys):
        dict_copy[key] = sim[i]
        folderName += f"{key}_{sim[i]}/"
    p = params.dict_to_class(dict_copy)
    p.concurrent_index = index
    p.folderName = folderName
    p.set_defaults()
    time_march(p)
    plotter.make_video(p)
    return folderName


if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        dict, p_init = params.load_file(f)

    # run simulations
    all_sims = list(product(*p_init.lists))

    folderNames = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(run_simulation, enumerate(all_sims)),
                total=len(all_sims),
                desc="Sim",
                leave=False,
            )
        )

    folderNames.extend(results)

    if len(all_sims) > 1:
        plotter.stack_videos(folderNames, dict["input_filename"], p_init.videos)
