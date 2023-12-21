import numpy as np


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
        pre_masked = False
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
            mask = np.random.rand(p.nx, p.ny, p.nm) > p.nu_fill
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


def set_boundary(s, X, Y, p):
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


def set_concentration(s, X, Y, p):
    # if hasattr(p, "temperature"):
    #     c = np.zeros_like(s)  # original bin that particles started in
    #     c[int(p.internal_geometry.perf_pts[0] * p.nx) : int(p.internal_geometry.perf_pts[1] * p.nx)] = 1
    #     c[int(p.internal_geometry.perf_pts[1] * p.nx) :] = 2
    #     c[np.isnan(s)] = np.nan
    if p.charge_discharge:
        if p.IC_mode == "full":
            c = np.ones_like(s)
        else:
            c = np.zeros_like(s)  # original bin that particles started in
            c[np.isnan(s)] = np.nan
    else:
        c = None

    return c
