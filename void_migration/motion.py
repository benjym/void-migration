import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import maximum_filter
import params
import operators
import random
from numba import jit


def stable_slope(
    i: int,
    j: int,
    dest: int,
    p: params.dict_to_class,
    nu: ArrayLike,
) -> bool:
    """Determine whether a void should swap with a solid particle.

    Args:
        i: an integer representing a row index
        j: an integer representing a column index
        lr: the cell we want to move into

    Returns:
        True if the void should NOT swap with a solid particle (i.e. the slope is stable). False otherwise.
    """
    if ((nu[dest, j] > nu[i, j]) and ((nu[dest, j] - nu[i, j]) < p.mu * p.nu_cs)) and (nu[i, j + 1] == 0):
        return True
    else:
        return False


def find_intersection(A, B):
    return np.array([x for x in A if x.tolist() in B.tolist()])


def delete_element(arr, element):
    return np.array([x for x in arr if not np.array_equal(x, element)])


# @jit(nopython=False)
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
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    # nu_max = maximum_filter(nu, footprint=kernel)  # , mode='constant', cval=0.0)
    # import matplotlib.pyplot as plt
    # plt.figure(7)
    # plt.subplot(211)
    # plt.imshow(nu.T)
    # plt.colorbar()
    # plt.subplot(212)
    # plt.imshow(nu_max.T)
    # plt.colorbar()
    # plt.pause(0.001)

    # dnu_dx, dnu_dy = np.gradient(nu)
    s_bar = operators.get_average(s)
    s_inv_bar = operators.get_hyperbolic_average(
        s
    )  # HACK: SHOULD RECALCULATE AFTER EVERY SWAP — WILL BE SUPER SLOW??
    # s_inv_bar[np.isnan(s_inv_bar)] = 1.0 / (1.0 / p.s_m + 1.0 / p.s_M)  # FIXME
    # s_bar[np.isnan(s_bar)] = (p.s_m + p.s_M)/2.  # FIXME

    scale_ang = (1 + (0.3 / (p.s_M / p.s_m))) * p.repose_angle / 90  ## Is there a grainsize effect?

    ###converting mean values to i,j,k format
    nu_req = np.repeat(nu[:, :], p.nm, axis=1).reshape(p.nx, p.ny, p.nm)
    s_inv_bar_req = np.repeat(s_inv_bar[:, :], p.nm, axis=1).reshape(p.nx, p.ny, p.nm)
    s_bar_req = np.repeat(s_bar[:, :], p.nm, axis=1).reshape(p.nx, p.ny, p.nm)

    P_initial = np.random.rand(p.nx, p.ny, p.nm)

    #########################################################################################################
    ##### UP
    # s_up = s.copy()
    # s_up = s_up[:,1:,:]

    P_ups = p.P_u_ref * (s_inv_bar_req[:, 1:, :] / s[:, 1:, :])
    # P_ups[np.isnan(P_ups)] = 0

    ############################### L #####################################################

    P_ls = p.P_lr_ref * (
        np.concatenate((s[[0]], s[0:-1, :, :]))[:, 0:-1, :]
        / np.concatenate((s_bar_req[[0]], s_bar_req[0:-1, :, :]))[:, 0:-1, :]
    )

    ###################################################### R #####################################################

    P_rs = p.P_lr_ref * (
        np.concatenate((s[1:, :, :], s[[-1]]))[:, 0:-1, :]
        / np.concatenate((s_bar_req[1:, :, :], s_bar_req[[-1]]))[:, 0:-1, :]
    )
    # P_rs[np.isnan(P_rs)] = 0

    ##############################################################################################################

    ids_up = np.where((nu_req[:, 0:-1, :] < p.nu_cs) & (np.isnan(s[:, 0:-1, :])) & (np.isnan(s[:, 1:, :])))
    if len(ids_up[0]) != 0:
        P_ups[ids_up] = 0  ## make probabilities 0 at satisfied condition if probability is NaN

    ids_left = np.where(
        (nu_req[:, 0:-1, :] < p.nu_cs)
        & (np.isnan(s[:, 0:-1, :]))
        & (np.isnan(np.concatenate((s[[0]], s[0:-1, :, :]))[:, 0:-1, :]))
    )
    if len(ids_left[0]) != 0:
        P_ls[ids_left] = 0  ## make probabilities 0 at satisfied condition if probability is NaN

    ids_right = np.where(
        (nu_req[:, 0:-1, :] < p.nu_cs)
        & (np.isnan(s[:, 0:-1, :]))
        & (np.isnan(np.concatenate((s[1:, :, :], s[[-1]]))[:, 0:-1, :]))
    )
    if len(ids_right[0]) != 0:
        P_rs[ids_right] = 0  ## make probabilities 0 at satisfied condition if probability is NaN

    P_ts = P_ups + P_ls + P_rs

    ids_up = np.where(
        (nu_req[:, 0:-1, :] < p.nu_cs)
        & (np.isnan(s[:, 0:-1, :]))
        & (~np.isnan(np.isnan(s[:, 1:, :])))
        & (P_initial[:, 0:-1, :] < (P_ups))
        & (P_ups > 0)
        & (P_ts > 0)
    )

    ids_swap_up = ids_up[0], ids_up[1] + 1, ids_up[2]

    ids_left = np.where(
        (nu_req[1:, 0:-1, :] < p.nu_cs)
        & (np.isnan(s[1:, 0:-1, :]))
        & (
            (~np.isnan(np.concatenate((s[[0]], s[0:-1, :, :]))[1:, 0:-1, :]))
            & (np.invert((((nu_req[0:-1, 0:-1, :] - nu_req[1:, 0:-1, :]) < scale_ang * p.nu_cs))))
        )
        & (P_initial[1:, 0:-1, :] < (P_ls[1:, :, :] + P_ups[1:, :, :]))
        & (P_ts[1:, :, :] > 0)
        & (P_ls[1:, :, :] > 0)
        & (P_initial[1:, 0:-1, :] >= P_ups[1:, :, :])
    )

    ids_swap_l = ids_left[0] + 1, ids_left[1], ids_left[2]

    # dx_r = nu_req[1:,0:-1,:] - nu_req[0:-1,0:-1,:]
    # dy_r = nu_req[0:-1,1:,:] - nu_req[0:-1,0:-1,:]

    # angle_r = np.abs(np.degrees(np.arctan(dx_r/dy_r)))
    # mag_r = np.sqrt(dx_r**2 + dy_r**2)

    ids_right = np.where(
        (nu_req[0:-1, 0:-1, :] < p.nu_cs)
        & (np.isnan(s[0:-1, 0:-1, :]))
        & (
            (~np.isnan(np.concatenate((s[1:, :, :], s[[-1]]))[0:-1, 0:-1, :]))
            & (np.invert((((nu_req[1:, 0:-1, :] - nu_req[0:-1, 0:-1, :]) < scale_ang * p.nu_cs))))
        )
        & (P_initial[0:-1, 0:-1, :] < (P_rs[0:-1, :, :] + P_ups[0:-1, :, :] + P_ls[0:-1, :, :]))
        & (P_ts[0:-1, :, :] > 0)
        & (P_rs[0:-1, :, :] > 0)
        & (P_initial[0:-1, 0:-1, :] >= (P_ups[0:-1, :, :] + P_ls[0:-1, :, :]))
    )

    ids_swap_r = ids_right[0] + 1, ids_right[1], ids_right[2]

    ## Destinations
    A = np.transpose(ids_swap_up)
    B = np.transpose(ids_left)
    C = np.transpose(ids_swap_r)

    # print("AAAAAAAAAAAAAAAAAAAA",A)
    # print("BBBBBBBBBBBBBBBBBBBB",B)
    # print("CCCCCCCCCCCCCCCCCCCC",C)

    # Handle A intersection B intersection C
    intersection = find_intersection(find_intersection(A, B), C)
    for selected_element in intersection:
        selected_array = random.choice([A, B, C])
        if np.array_equal(selected_array, A):
            B = delete_element(B, selected_element)
            C = delete_element(C, selected_element)
        elif np.array_equal(selected_array, B):
            A = delete_element(A, selected_element)
            C = delete_element(C, selected_element)
        else:
            A = delete_element(A, selected_element)
            B = delete_element(B, selected_element)

    # Handle A intersection B
    intersection = find_intersection(A, B)
    for selected_element in intersection:
        selected_array = random.choice([A, B])
        if np.array_equal(selected_array, A):
            B = delete_element(B, selected_element)
        else:
            A = delete_element(A, selected_element)

    # Handle B intersection C
    intersection = find_intersection(B, C)
    for selected_element in intersection:
        selected_array = random.choice([B, C])
        if np.array_equal(selected_array, B):
            C = delete_element(C, selected_element)
        else:
            B = delete_element(B, selected_element)

    # Handle A intersection C
    intersection = find_intersection(A, C)
    for selected_element in intersection:
        selected_array = random.choice([A, C])
        if np.array_equal(selected_array, A):
            C = delete_element(C, selected_element)
        else:
            A = delete_element(A, selected_element)

    # print("A:", A,len(A) > 0)
    # print("B:", B)
    # print("C:", C)
    # print("AAAAAAAAAAAAAAAAAAAA",A)

    A_ori = (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    B_ori = (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    C_ori = (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64))

    if len(A) > 0:
        A_ori = tuple(np.transpose(A))
        A_ori = A_ori[0], A_ori[1] - 1, A_ori[2]  # Source
        A = tuple(np.transpose(A))  # Destination
    else:
        A = (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64))

    if len(B) > 0:
        B_ori = tuple(np.transpose(B))
        B_ori = B_ori[0] + 1, B_ori[1], B_ori[2]  # Source
        B = tuple(np.transpose(B))  # Destination
    else:
        B = (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64))

    if len(C) > 0:
        C_ori = tuple(np.transpose(C))
        C_ori = C_ori[0] - 1, C_ori[1], C_ori[2]  # Source
        C = tuple(np.transpose(C))  # Destination
    else:
        C = (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64))

    all_ids = (
        np.hstack((A_ori[0], B_ori[0], C_ori[0])),
        np.hstack((A_ori[1], B_ori[1], C_ori[1])),
        np.hstack((A_ori[2], B_ori[2], C_ori[2])),
    )
    all_swap_ids = (
        np.hstack((A[0], B[0], C[0])),
        np.hstack((A[1], B[1], C[1])),
        np.hstack((A[2], B[2], C[2])),
    )

    s[all_ids], s[all_swap_ids] = s[all_swap_ids], s[all_ids]
    c[all_ids], c[all_swap_ids] = c[all_swap_ids], c[all_ids]

    # T[ids_left],T[ids_swap] = T[ids_swap],T[ids_left]

    nu_req[all_ids] += 1 / p.nm
    nu_req[all_swap_ids] -= 1 / p.nm

    nu = nu_req[:, :, 0]

    return u, v, s, c, T, N_swap


def add_voids(u, v, s, p, c, outlet):
    if p.add_voids == "central_outlet":  # Remove at central outlet - use this one
        # print("UUUUUUUUUUUUUUUUUUU",p.nx // 2 - p.half_width, p.nx // 2 + p.half_width + 1)
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
    elif p.add_voids == "wall":  # Remove at central outlet - use this one
        for i in range(0, p.half_width):
            for k in range(p.nm):
                # if np.random.rand() < 0.1:
                if not np.isnan(s[i, 0, k]):
                    if p.refill:
                        if np.sum(np.isnan(s[0 : p.half_width, -1, k])) > 0:
                            target = np.random.choice(np.nonzero(np.isnan(s[0 : p.half_width, -1, k]))[0])
                            s[target, -1, k], s[i, 0, k] = s[i, 0, k], s[target, -1, k]

                    else:
                        s[i, 0, k] = np.nan
                        if hasattr(p, "charge_discharge"):
                            c[i, 0, k] = np.nan
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
            if p.silo_width == "half":
                x_points = np.arange(0, p.half_width)
                req = np.random.choice(
                    [p.s_m, p.s_M], size=[p.half_width, p.nm]
                )  # create an array of grainsizes

                mask = np.random.rand(p.half_width, p.nm) > p.fill_ratio
                req[mask] = np.nan

            elif p.silo_width == "full":
                x_points = np.arange(p.nx // 2 - p.half_width, p.nx // 2 + p.half_width + 1)

                req = np.random.choice(
                    [p.s_m, p.s_M], size=[(p.nx // 2 + p.half_width + 1) - (p.nx // 2 - p.half_width), p.nm]
                )  # create an array of grainsizes
                mask = (
                    np.random.rand((p.nx // 2 + p.half_width + 1) - (p.nx // 2 - p.half_width), p.nm)
                    > p.fill_ratio
                )  # create how much to fill
                req[mask] = np.nan  # convert some cells to np.nan
        if p.gsd_mode == "fbi":  # bidisperse
            if p.silo_width == "half":
                x_points = np.arange(0, p.half_width)
                #     req = np.random.choice(
                #     [p.s_m, p.Fr*p.s_m, p.s_M, p.Fr*p.s_M], size=[p.half_width, p.nm]
                # )  # create an array of grainsizes
                f_1 = p.half_width - int(p.half_width / 2)
                f_2 = p.half_width - f_1
                req1 = np.random.uniform(p.s_m, p.Fr * p.s_m, size=[p.nm, f_1])
                req2 = np.random.uniform(p.s_M, p.Fr * p.s_M, size=[p.nm, f_2])
                # print("1111111111111111111",np.shape(req1))
                # print("2222222222222222222",np.shape(req2))
                req3 = np.concatenate((req1, req2), axis=1)
                # print("3333333333333333333",np.shape(req3))
                req = req3.reshape(p.half_width, p.nm)
                # print("4444444444444444444",np.shape(req))
                mask = np.random.rand(p.half_width, p.nm) > p.fill_ratio
                req[mask] = np.nan
                # print("YYYYYYYYYYYYYYYYYYYYYY",np.sum(~np.isnan(req)), np.nanmin(req),np.nanmax(req))

            elif p.silo_width == "full":
                x_points = np.arange(p.nx // 2 - p.half_width, p.nx // 2 + p.half_width)

                f_1 = int(len(x_points) - int(len(x_points) / 2))
                f_2 = int(len(x_points) - f_1)

                req1 = np.random.uniform(p.s_m, p.Fr * p.s_m, size=[p.nm, f_1])
                req2 = np.random.uniform(p.s_M, p.Fr * p.s_M, size=[p.nm, f_2])

                req3 = np.concatenate((req1, req2), axis=1)

                req = req3.reshape(int(len(x_points)), p.nm)

                mask = (
                    np.random.rand((p.nx // 2 + p.half_width) - (p.nx // 2 - p.half_width), p.nm)
                    > p.fill_ratio
                )  # create how much to fill
                req[mask] = np.nan  # convert some cells to np.nan
                # print("YYYYYYYYYYYYYYYYYYYYYY",np.sum(~np.isnan(req)), np.nanmin(req),np.nanmax(req), f_1, f_2, np.shape(req), np.shape(mask))

        if p.gsd_mode == "mono":
            if p.silo_width == "half":
                x_points = np.arange(0, p.half_width)
                req = p.s_m * np.ones([p.half_width, p.nm])  # monodisperse

                p.s_M = p.s_m
                mask = np.random.rand(p.half_width, p.nm) > p.fill_ratio
                req[mask] = np.nan

            elif p.silo_width == "full":
                x_points = np.arange(p.nx // 2 - p.half_width, p.nx // 2 + p.half_width + 1)
                # print("UUUUUUUUUUUU",x_points)
                req = p.s_m * np.ones(
                    [(p.nx // 2 + p.half_width + 1) - (p.nx // 2 - p.half_width), p.nm]
                )  # monodisperse

                p.s_M = p.s_m
                mask = (
                    np.random.rand((p.nx // 2 + p.half_width + 1) - (p.nx // 2 - p.half_width), p.nm)
                    > p.fill_ratio
                )
                req[mask] = np.nan
                # print("IIIIIIIIIIIIIIIIIIIIIII",np.count_nonzero(req[~np.isnan(req)]))

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
        # print("JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",np.count_nonzero(req[~np.isnan(req)]))
        # print("JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",np.count_nonzero(s[~np.isnan(s)]),np.count_nonzero(~np.isnan(s)))
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
