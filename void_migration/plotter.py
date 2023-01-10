import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import warnings

plt.inferno()
cdict = {
    "red": ((0.0, 1.0, 1.0), (0.25, 1.0, 1.0), (0.5, 1.0, 1.0), (0.75, 0.902, 0.902), (1.0, 0.0, 0.0)),
    "green": (
        (0.0, 0.708, 0.708),
        (0.25, 0.302, 0.302),
        (0.5, 0.2392, 0.2392),
        (0.75, 0.1412, 0.1412),
        (1.0, 0.0, 0.0),
    ),
    "blue": (
        (0.0, 0.4, 0.4),
        (0.25, 0.3569, 0.3569),
        (0.5, 0.6078, 0.6078),
        (0.75, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ),
}
orange_blue_cmap = LinearSegmentedColormap("grainsize", cdict, 256)
orange_blue_cmap.set_bad("w", 1.0)
orange_blue_cmap.set_under("w", 1.0)
grey = cm.get_cmap("gray")
grey.set_bad("w", 0.0)
bwr = cm.get_cmap("bwr")
bwr.set_bad("k", 1.0)


def plot_u_time(y, U, nu_time, folderName, nt):
    U = np.ma.masked_where(nu_time < 0.2, U)
    U = np.amax(U) - U
    # fig = plt.figure()

    plt.pcolormesh(U.T)  # ,cmap='bwr',vmin=-amax(abs(U))/2,vmax=amax(abs(U))/2)
    plt.colorbar()
    plt.savefig(folderName + "u_time.png")

    plt.clf()
    u_y = np.mean(U[nt // 2 :], 0)
    # gamma_dot_y = gradient(u_y,dy)
    # plt.plot(gamma_dot_y,y,'r')
    plt.plot(u_y, y, "b")
    plt.savefig(folderName + "u_avg.png")
    np.save(folderName + "u_y.npy", np.ma.filled(u_y, np.nan))
    np.save(folderName + "nu.npy", np.mean(nu_time[nt // 2 :], axis=0))


def plot_s_bar(s_bar, nu_time, s_m, folderName):
    # plt.figure()

    plt.clf()
    plt.pcolormesh(s_bar.T, cmap=orange_blue_cmap, vmin=s_m, vmax=1)
    plt.colorbar()
    plt.savefig(folderName + "s_bar.png")
    np.save(folderName + "s_bar.npy", s_bar.T)

    plt.clf()
    plt.pcolormesh(nu_time.T, cmap="inferno", vmin=0, vmax=1)
    plt.colorbar()
    plt.savefig(folderName + "nu.png")


def plot_s(x, y, s, folderName, t, internal_geometry, s_m):
    plt.clf()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        s_plot = np.nanmean(s, axis=2).T
    s_plot = np.ma.masked_where(np.isnan(s_plot), s_plot)
    plt.pcolormesh(x, y, s_plot, cmap=orange_blue_cmap, vmin=s_m, vmax=1)
    # plt.colorbar()
    if internal_geometry:
        for i in internal_geometry["perf_pts"]:
            plt.plot([x[i], x[i]], [y[0], y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(folderName + "s_" + str(t).zfill(6) + ".png", dpi=100)


def plot_nu(x, y, s, folderName, t, internal_geometry, boundary=False):
    nu = 1 - np.mean(np.isnan(s), axis=2).T
    if internal_geometry:
        nu = np.ma.masked_where(boundary.T, nu)
    plt.clf()
    plt.pcolormesh(x, y, nu, cmap="inferno", vmin=0, vmax=1)
    if internal_geometry:
        if internal_geometry["perf_plate"]:
            for i in internal_geometry["perf_pts"]:
                plt.plot([x[i], x[i]], [y[0], y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(folderName + "nu_" + str(t).zfill(6) + ".png", dpi=100)


def plot_u(x, y, s, u, v, folderName, t, nm, IC_mode, internal_geometry, boundary):
    # mask = mean(isnan(s),axis=2) > 0.95
    # u = ma.masked_where(mask,u/sum(isnan(s),axis=2)).T
    # v = ma.masked_where(mask,v/sum(isnan(s),axis=2)).T
    u = u.T
    v = v.T

    if IC_mode == "slope":
        u = np.amax(u) - u  # lagrangian

    plt.clf()
    # plt.quiver(X,Y,u,v)
    # print(u)
    plt.pcolormesh(x, y, u, vmin=-np.amax(np.abs(u)), vmax=np.amax(np.abs(u)), cmap="bwr")
    if internal_geometry:
        if internal_geometry["perf_plate"]:
            for i in internal_geometry["perf_pts"]:
                plt.plot([x[i], x[i]], [y[0], y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    # plt.colorbar()
    plt.savefig(folderName + "u_" + str(t).zfill(6) + ".png", dpi=100)

    plt.clf()
    # plt.quiver(X,Y,u,v)
    plt.pcolormesh(x, y, v, vmin=-np.amax(np.abs(v)), vmax=np.amax(np.abs(v)), cmap="bwr")
    if internal_geometry:
        if internal_geometry["perf_plate"]:
            for i in internal_geometry["perf_pts"]:
                plt.plot([x[i], x[i]], [y[0], y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(folderName + "v_" + str(t).zfill(6) + ".png", dpi=100)

    U = np.sqrt(u**2 + v**2)
    plt.clf()
    plt.pcolormesh(x, y, np.ma.masked_where(boundary.T, U), vmin=0, vmax=np.amax(np.abs(U)), cmap="inferno")
    if internal_geometry:
        if internal_geometry["perf_plate"]:
            for i in internal_geometry["perf_pts"]:
                plt.plot([x[i], x[i]], [y[0], y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(folderName + "U_mag_" + str(t).zfill(6) + ".png", dpi=100)


def plot_c(x, y, s, c, folderName, t, internal_geometry):
    nm = s.shape[2]
    mask = np.sum(np.isnan(s), axis=2) > 0.95 * nm

    plt.clf()
    plt.pcolormesh(x, y, np.ma.masked_where(mask, np.nanmean(c, axis=2)).T, cmap="inferno", vmin=0, vmax=2)
    if internal_geometry:
        if internal_geometry["perf_plate"]:
            for i in internal_geometry["perf_pts"]:
                plt.plot([x[i], x[i]], [y[0], y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(folderName + "c_" + str(t).zfill(6) + ".png", dpi=100)


def plot_outlet(outlet, folderName):
    plt.clf()
    plt.plot(outlet)
    plt.xlabel("time")
    plt.ylabel("outflow")
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    plt.savefig(folderName + "outflow.png", dpi=100)


def plot_profile(x, nu_time_x, folderName, nt, t_f):
    plt.clf()
    plt.pcolormesh(x, np.linspace(0, t_f, nt), nu_time_x)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    plt.savefig(folderName + "collapse_profile.png", dpi=100)


def plot_T(x, y, s, T, folderName, t, boundary_temperature, inlet_temperature):
    nm = s.shape[2]
    mask = np.sum(np.isnan(s), axis=2) > 0.95 * nm
    plt.clf()
    plt.pcolormesh(
        x,
        y,
        np.ma.masked_where(mask, np.nanmean(T, axis=2)).T,
        cmap=bwr,
        vmin=boundary_temperature,
        vmax=inlet_temperature,
    )
    plt.axis("off")
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(folderName + "T_" + str(t).zfill(6) + ".png", dpi=100)
