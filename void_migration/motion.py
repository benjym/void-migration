import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import maximum_filter
import params
import operators

# from numba import njit


def stable_slope_fast(s, dir, delta_limit):
    dest = np.roll(s, dir, axis=0)
    nu_here = operators.get_solid_fraction(s)
    # nu_dest = operators.get_solid_fraction(dest)
    nu_dest = np.roll(nu_here, dir, axis=0)
    delta_nu = nu_dest - nu_here

    stable = delta_nu <= delta_limit
    Stable = np.repeat(stable[:, :, np.newaxis], s.shape[2], axis=2)
    return Stable


def stable_slope(s, i, j, dest, p):
    """
    Determines if the slope between two points is stable based on the solid fraction difference.

    Parameters:
    - s (object): The simulation state, containing the grid and relevant data.
    - i (int): The current row index in the grid.
    - j (int): The current column index in the grid.
    - dest (int): The destination row index for comparison.
    - p (object): An object containing simulation parameters, including the delta_limit.

    Returns:
    - bool: True if the difference in solid fraction between the current point and the destination
        point is less than or equal to the delta_limit, indicating a stable slope. False otherwise.

    This function calculates the solid fraction at the current point (i, j) and at a destination point
    (dest, j), then compares the difference in solid fraction to a threshold (delta_limit) defined in
    the parameter object p. If the difference is less than or equal to the threshold, the function
    returns True, indicating the slope is stable. Otherwise, it returns False.
    """
    nu_here = operators.get_solid_fraction(s, [i, j])
    nu_dest = operators.get_solid_fraction(s, [dest, j])
    delta_nu = nu_dest - nu_here

    return delta_nu <= p.delta_limit


def locally_solid(s, i, j, p):
    """
    Determines if a given point in the simulation grid is locally solid based on the solid fraction threshold.

    Parameters:
    - s (object): The simulation state, containing the grid and relevant data.
    - i (int): The row index of the point in the grid.
    - j (int): The column index of the point in the grid.
    - p (object): An object containing simulation parameters, including the critical solid fraction threshold (nu_cs).

    Returns:
    - bool: True if the solid fraction at the given point is greater than or equal to the critical solid fraction threshold (nu_cs), indicating the point is locally solid. False otherwise.

    This function calculates the solid fraction at the specified point (i, j) in the simulation grid. It then compares this value to the critical solid fraction threshold (nu_cs) defined in the parameter object p. If the solid fraction at the point is greater than or equal to nu_cs, the function returns True, indicating the point is considered locally solid. Otherwise, it returns False.
    """
    nu = operators.get_solid_fraction(s, [i, j])
    return nu >= p.nu_cs


def empty_nearby(nu, p):
    """
    Identifies empty spaces adjacent to each point in a grid based on a given solid fraction matrix.

    Parameters:
    - nu (numpy.ndarray): A 2D array representing the solid fraction at each point in the grid.
    - p (object): An object containing simulation parameters, not used in this function but included for consistency with the interface.

    Returns:
    - numpy.ndarray: A boolean array where True indicates an empty space adjacent to the corresponding point in the input grid.

    This function applies a maximum filter with a cross-shaped kernel to the solid fraction matrix 'nu'. The kernel is defined to consider the four cardinal directions (up, down, left, right) adjacent to each point. The maximum filter operation identifies the maximum solid fraction value in the neighborhood defined by the kernel for each point. Points where the maximum solid fraction in their neighborhood is 0 are considered adjacent to an empty space, and the function returns a boolean array marking these points as True.
    """
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    nu_max = maximum_filter(nu, footprint=kernel)  # , mode='constant', cval=0.0)

    return nu_max == 0


# @njit
def move_voids_fast(
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

    for axis, d in [(1, -1), (0, -1), (0, 1)]:  # up, left, right
        nu = 1.0 - np.mean(np.isnan(s), axis=2)

        solid = nu >= p.nu_cs
        Solid = np.repeat(solid[:, :, np.newaxis], p.nm, axis=2)

        unstable = np.isnan(s) * ~Solid  # & ~Skip

        dest = np.roll(s, d, axis=axis)

        if axis == 1:
            s_inv_bar = operators.get_hyperbolic_average(s)
            S_inv_bar = np.repeat(s_inv_bar[:, :, np.newaxis], p.nm, axis=2)
            S = np.roll(S_inv_bar, d, axis=axis)
            P = p.P_u_ref * (S / dest)

            P[:, -1, :] = 0  # no swapping up from top row
        elif axis == 0:
            s_bar = operators.get_average(s)
            S_bar = np.repeat(s_bar[:, :, np.newaxis], p.nm, axis=2)
            S = np.roll(S_bar, d, axis=axis)
            P = p.P_lr_ref * (dest / S)

            if d == -1:  # left
                P[0, :, :] = 0  # no swapping left from leftmost column
            elif d == 1:  # right
                P[-1, :, :] = 0  # no swapping right from rightmost column

            slope_stable = stable_slope_fast(s, d, p.delta_limit)
            P[slope_stable] = 0
            # P *= stable_slope_fast(s, d, p.delta_limit)

        swap_possible = unstable * ~np.isnan(dest)
        P = np.where(swap_possible, P, 0)
        swap = np.random.rand(*P.shape) < P

        total_swap = np.sum(swap, axis=2, dtype=int)
        max_swap = ((p.nu_cs - nu) * p.nm).astype(int)

        overfilled = total_swap - max_swap
        overfilled = np.maximum(overfilled, 0)

        for i in range(p.nx):
            for j in range(p.ny):
                if overfilled[i, j] > 0:
                    swap_args = np.argwhere(swap[i, j, :]).flatten()
                    over_indices = np.random.choice(swap_args, size=overfilled[i, j], replace=False)
                    swap[i, j, over_indices] = False

        swap_indices = np.argwhere(swap)
        dest_indices = swap_indices.copy()
        dest_indices[:, axis] -= d

        if axis == 1:
            v[swap_indices[:, 0], swap_indices[:, 1]] += d
        elif axis == 0:
            u[swap_indices[:, 0], swap_indices[:, 1]] += d

        (
            s[swap_indices[:, 0], swap_indices[:, 1], swap_indices[:, 2]],
            s[dest_indices[:, 0], dest_indices[:, 1], dest_indices[:, 2]],
        ) = (
            s[dest_indices[:, 0], dest_indices[:, 1], dest_indices[:, 2]],
            s[swap_indices[:, 0], swap_indices[:, 1], swap_indices[:, 2]],
        )

    return u, v, s, c, T, N_swap


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

    nu = 1.0 - np.mean(np.isnan(s), axis=2)

    skip = empty_nearby(nu, p)

    s_bar = operators.get_average(s)
    s_inv_bar = operators.get_hyperbolic_average(s)

    for index in p.indices:
        i, j, k = np.unravel_index(index, [p.nx, p.ny - 1, p.nm])

        if not skip[i, j]:
            if np.isnan(s[i, j, k]):
                if not locally_solid(s, i, j, p):
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

                    if np.isnan(s[l, j + diag, k]) or stable_slope(s, i, j, l, p):
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

                    if np.isnan(s[r, j + diag, k]) or stable_slope(s, i, j, r, p):
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

                    if P_tot > 1:
                        print(f"Error: P_tot > 1, P_u = {P_u}, P_l = {P_l}, P_r = {P_r}")

                    dest = None
                    if P_tot > 0:
                        P = np.random.rand()
                        if P < P_u and P_u > 0:  # go up
                            dest = [i, j + 1, k]
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

    elif p.add_voids == "place_on_top":  # pour in centre starting at base
        if p.gsd_mode == "bi":  # bidisperse
            x_points = np.arange(p.nx // 2 - p.half_width, p.nx // 2 + p.half_width + 1)

            req = np.random.choice(
                [p.s_m, p.s_M], size=[(p.nx // 2 + p.half_width + 1) - (p.nx // 2 - p.half_width), p.nm]
            )  # create an array of grainsizes
            mask = (
                np.random.rand((p.nx // 2 + p.half_width + 1) - (p.nx // 2 - p.half_width), p.nm)
                > p.fill_ratio
            )  # create how much to fill
            req[mask] = np.nan  # convert some cells to np.nan
        if p.gsd_mode == "mono":
            x_points = np.arange(p.nx // 2 - p.half_width, p.nx // 2 + p.half_width + 1)
            req = p.s_m * np.ones(
                [(p.nx // 2 + p.half_width + 1) - (p.nx // 2 - p.half_width), p.nm]
            )  # monodisperse

            p.s_M = p.s_m
            mask = (
                np.random.rand((p.nx // 2 + p.half_width + 1) - (p.nx // 2 - p.half_width), p.nm)
                > p.fill_ratio
            )
            req[mask] = np.nan

        den = 1 - np.mean(np.isnan(s), axis=2)
        if np.mean(den) == 0.0:
            for i in range(len(x_points)):
                for k in range(p.nm):
                    s[x_points[i], 0, k] = req[i, k]
                    if ~np.isnan(req[i, k]):
                        c[x_points[i], 0, k] = p.current_cycle
        else:
            for i in range(len(x_points)):
                for k in range(p.nm):
                    if (
                        np.isnan(s[x_points[i], 0, k])
                        and np.count_nonzero(np.isnan(s[x_points[i], :, k])) == p.ny
                    ):
                        s[x_points[i], 0, k] = req[i, k]
                        if ~np.isnan(req[i, k]):
                            c[x_points[i], 0, k] = p.current_cycle
                    else:
                        a = np.max(np.argwhere(~np.isnan(s[x_points[i], :, k])))  # choose the max ht
                        if a >= p.ny - 2:
                            pass
                        else:
                            s[x_points[i], a + 1, k] = req[i, k]  # place a cell on the topmost cell "a+1"
                            if ~np.isnan(req[i, k]):
                                c[x_points[i], a + 1, k] = p.current_cycle

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


def close_voids(u, v, s, p):
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
