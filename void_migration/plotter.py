import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import warnings
import operators

# _video_encoding = ["-c:v", "libx265", "-preset", "fast", "-crf", "28", "-tag:v", "hvc1"] # nice small file sizes
_video_encoding = [
    "-c:v",
    "libx264",
    "-preset",
    "slow",
    "-profile:v",
    "high",
    "-level:v",
    "4.0",
    "-pix_fmt",
    "yuv420p",
    "-crf",
    "22",
]  # powerpoint compatible

_dpi = 20  # *10
# plt.rcParams["figure.dpi"] = _dpi


def is_ffmpeg_installed():
    try:
        result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0
    except FileNotFoundError:
        return False


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
colors = [(1, 0, 0), (0, 0, 1)]
cmap_name = "my_list"
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)
inferno = cm.get_cmap("inferno")
inferno.set_bad("w", 0.0)
inferno_r = cm.get_cmap("inferno_r")
inferno_r.set_bad("w", 0.0)
Dark2 = cm.get_cmap("Dark2")

global fig, summary_fig


def set_plot_size(p):
    global fig, summary_fig

    # wipe any existing figures
    for i in plt.get_fignums():
        plt.close(i)
    fig = plt.figure(figsize=[p.nx / _dpi, p.ny / _dpi])
    summary_fig = plt.figure()


def update(x, y, s, u, v, c, T, outlet, p, t, *args):
    if "s" in p.plot:
        plot_s(x, y, s, p, t)
    if "diff_s" in p.plot:
        plot_diff_s(x, y, s, p, t)
    if "nu" in p.plot:
        plot_nu(x, y, s, p, t)
    if "rel_nu" in p.plot:
        plot_relative_nu(x, y, s, p, t)
    if "U_mag" in p.plot:
        plot_u(x, y, s, u, v, p, t)
    if "c" in p.plot:
        plot_c(x, y, s, c, p, t)
    if "temperature" in p.plot:
        plot_T(x, y, s, T, p, t)
    # if "density_profile" in p.plot:
    #     plot_profile(x, nu_time_x, p)
    if "permeability" in p.plot:
        plot_permeability(x, y, s, p, t)

    if "s" in p.save:
        save_s(x, y, s, p, t)
    if "nu" in p.save:
        save_nu(x, y, s, p, t)
    if "rel_nu" in p.save:
        save_relative_nu(x, y, s, p, t)
    # if "U_mag" in p.save:
    #     save_u(x, y, s, u, v, p, t)
    if "permeability" in p.save:
        save_permeability(x, y, s, p, t)
    if "concentration" in p.save:
        save_c(c, p.folderName, t)
    if "outlet" in p.save:
        np.savetxt(p.folderName + "outlet.csv", np.array(outlet), delimiter=",")
    # if "temperature" in p.save:
    #     np.savetxt(p.folderName + "outlet_T.csv", np.array(outlet_T), delimiter=",")
    if "velocity" in p.save:
        np.savetxt(p.folderName + "u.csv", u / np.sum(np.isnan(s), axis=2), delimiter=",")
    if "charge_discharge" in p.save:
        c_d_saves(p, non_zero_nu_time, p_count, p_count_s, p_count_l)


def plot_u_time(y, U, nu_time, p):
    plt.figure(summary_fig)

    U = np.ma.masked_where(nu_time < 0.2, U)
    # U = np.amax(U) - U

    plt.clf()
    plt.pcolormesh(U.T)  # ,cmap='bwr',vmin=-amax(abs(U))/2,vmax=amax(abs(U))/2)
    plt.colorbar()
    plt.savefig(p.folderName + "u_time.png")

    plt.clf()
    u_y = np.mean(U[p.nt // 2 :], 0)
    # gamma_dot_y = gradient(u_y,dy)
    # plt.plot(gamma_dot_y,y,'r')
    plt.plot(u_y, y, "b")
    plt.xlabel("Average horizontal velocity (m/s)")
    plt.ylabel("Height (m)")
    plt.savefig(p.folderName + "u_avg.png")

    np.save(p.folderName + "u_y.npy", np.ma.filled(u_y, np.nan))
    np.save(p.folderName + "nu.npy", np.mean(nu_time[p.nt // 2 :], axis=0))


def plot_pdf_cdf(p, s, xpoints, ypoints, t):
    # if p.gsd_mode = 'mono' or p.gsd_mode = 'bi':
    pdfs = []
    cdfs = []

    pdfs_log = []
    cdfs_log = []

    if p.gsd_mode == "fbi":
        binn = np.linspace(p.s_m - (p.s_m / 10), (p.Fr * p.s_M) + ((p.Fr * p.s_M) / 10), int(p.nm / 5))
        binn_log = np.logspace(
            np.log10(p.s_m - (p.s_m / 10)), np.log10((p.Fr * p.s_M) + ((p.Fr * p.s_M) / 10)), int(p.nm / 5)
        )

    else:
        binn = np.linspace(p.s_m - (p.s_m / 10), p.s_M + (p.s_M / 10), int(p.nm / 5))
        binn_log = np.logspace(np.log10(p.s_m - (p.s_m / 10)), np.log10(p.s_M + (p.s_M / 10)), int(p.nm / 5))

    for i in xpoints:
        pdfsi = []
        cdfsi = []

        pdfs_logi = []
        cdfs_logi = []
        for j in ypoints:
            nu = 1 - np.mean(np.isnan(s[i, j, :]))
            if np.count_nonzero(np.isnan(s[i, j, :])) == p.nm or nu < p.nu_cs / 7.0:
                pdf = np.zeros(len(binn) - 1)
                cdf = np.zeros(len(binn) - 1)
                pdf_log = np.zeros(len(binn_log) - 1)
                cdf_log = np.zeros(len(binn_log) - 1)
            else:
                count, bins_count = np.histogram(s[i, j, :], bins=binn)
                count_log, bins_count_log = np.histogram(s[i, j, :], bins=binn_log)
                pdf = count / np.sum(count)
                cdf = np.cumsum(pdf)
                pdf_log = count_log / np.sum(count_log)
                cdf_log = np.cumsum(pdf_log)

            pdfsi.append(pdf)
            cdfsi.append(cdf)
            pdfs_logi.append(pdf_log)
            cdfs_logi.append(cdf_log)

        pdfs.append(pdfsi)
        cdfs.append(cdfsi)
        pdfs_log.append(pdfs_logi)
        cdfs_log.append(cdfs_logi)

    np.save(p.folderName + "pdf_" + str(t).zfill(6) + ".npy", pdfs)
    np.save(p.folderName + "cdf_" + str(t).zfill(6) + ".npy", cdfs)
    np.save(p.folderName + "pdf_log_" + str(t).zfill(6) + ".npy", pdfs_log)
    np.save(p.folderName + "cdf_log_" + str(t).zfill(6) + ".npy", cdfs_log)


def plot_s_bar(y, s_bar, nu_time, p):
    plt.figure(summary_fig)

    if p.mask_s_bar:
        masked_s_bar = np.ma.masked_where(nu_time < p.nu_cs / 10.0, s_bar)
    else:
        masked_s_bar = s_bar

    plt.clf()
    plt.pcolormesh(
        np.linspace(0, p.t_f, p.nt), y, masked_s_bar.T, cmap=orange_blue_cmap, vmin=p.s_m, vmax=p.s_M
    )
    plt.colorbar()
    plt.xlabel("Time (s)")
    plt.ylabel("Height (m)")
    plt.savefig(p.folderName + "s_bar.png")
    np.save(p.folderName + "s_bar.npy", s_bar.T)

    plt.clf()
    plt.pcolormesh(np.linspace(0, p.t_f, p.nt), y, nu_time.T, cmap=inferno, vmin=0, vmax=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Height (m)")
    plt.colorbar()
    plt.savefig(p.folderName + "nu.png")


def save_coordinate_system(x, y, p):
    np.savetxt(p.folderName + "x.csv", x, delimiter=",")
    np.savetxt(p.folderName + "y.csv", y, delimiter=",")


def c_d_saves(p, non_zero_nu_time, *args):
    np.save(p.folderName + "nu_non_zero_avg.npy", non_zero_nu_time)
    if p.gsd_mode == "mono":
        np.save(p.folderName + "cell_count.npy", args[0])
    elif p.gsd_mode == "bi":
        np.save(p.folderName + "cell_count.npy", args[0])
        np.save(p.folderName + "cell_count_s.npy", args[1])
        np.save(p.folderName + "cell_count_l.npy", args[2])


def kozeny_carman(s):
    sphericity = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        porosity = np.mean(np.isnan(s), axis=2)
        s_bar = operators.get_average(s)
        permeability = sphericity**2 * (porosity**3) * s_bar**2 / (180 * (1 - porosity) ** 2)
    return permeability


def plot_permeability(x, y, s, p, t):
    """
    Calculate and save the permeability of the domain at time t.
    """
    permeability = kozeny_carman(s)

    plt.figure(fig)
    plt.clf()
    plt.pcolormesh(x, y, permeability.T, cmap=inferno)
    plt.axis("off")
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(p.folderName + "permeability_" + str(t).zfill(6) + ".png")


def save_permeability(x, y, s, p, t):
    permeability = kozeny_carman(s)
    np.savetxt(p.folderName + "permeability_" + str(t).zfill(6) + ".csv", permeability, delimiter=",")


def plot_s(x, y, s, p, t):
    plt.figure(fig)
    plt.clf()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        s_plot = np.nanmean(s, axis=2).T
        # s_plott = np.nanmean(s, axis=2)
    # # s_plot = np.ma.masked_where(np.isnan(s_plot), s_plot)
    nu = 1 - np.mean(np.isnan(s), axis=2).T
    # s_plot = np.ma.masked_where(nu < p.nu_cs / 7.0, s_plot)   ##uncomment when masking low solid fraction values
    # # nu[nu > p.nu_cs] = p.nu_cs

    # # alpha_vals = 1 - np.exp(-10*nu)

    # # print("********************",np.max(nu),np.min(nu),np.max(alpha_vals),np.min(alpha_vals))

    if p.gsd_mode == "fbi":
        plt.pcolormesh(
            x,
            y,
            s_plot,
            cmap=orange_blue_cmap,
            vmin=p.s_m - (p.s_m / 100),
            vmax=(p.Fr * p.s_M) + ((p.Fr * p.s_M) / 100),
        )
    else:
        plt.pcolormesh(
            x, y, s_plot, cmap=orange_blue_cmap, vmin=p.s_m - (p.s_m / 100), vmax=p.s_M + (p.s_M / 100)
        )
    # plt.pcolormesh(x, y, s_plot,cmap=orange_blue_cmap, vmin=np.min(s)-(np.min(s)/100), vmax=np.max(s)+(np.max(s)/100))

    if p.internal_geometry:
        for i in p.internal_geometry["perf_pts"]:
            plt.plot([x[i], x[i]], [y[0], y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    if p.gsd_mode == "fbi":
        ticks = np.linspace(p.s_m, p.s_M * p.Fr, 3, endpoint=True)
    else:
        ticks = np.linspace(p.s_m, p.s_M, 3, endpoint=True)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if hasattr(p, "plot_colorbar"):
        plt.colorbar(shrink=0.5, location="top", pad=0.01, ticks=ticks, format="%2.1e")
        # np.save(p.folderName + "s_" + str(t).zfill(6) + ".npy",s_plott)
    plt.savefig(
        p.folderName + "s_" + str(t).zfill(6) + ".png"
    )  # ,dpi = 120/1.5)#,dpi = _dpi)#, bbox_inches="tight", dpi=100)


def plot_diff_s(x, y, s, p, t):
    plt.figure(fig)
    plt.clf()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        s_plot = np.sum(np.isnan(s), axis=2).T / p.nm
        s_tmp_plot = np.sum(np.isnan(p.tmp_s), axis=2).T / p.nm
    s_plot = s_plot - s_tmp_plot

    plt.pcolormesh(x, y, s_plot, cmap="gray_r", vmin=0, vmax=0.125)  # ,linewidth=0,rasterized = True)
    if p.internal_geometry:
        for i in p.internal_geometry["perf_pts"]:
            plt.plot([x[i], x[i]], [y[0], y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    ticks = np.linspace(0, 0.125, 3, endpoint=True)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if hasattr(p, "plot_colorbar"):
        plt.colorbar(shrink=0.5, location="top", pad=0.01, ticks=ticks)  # , format="%2.1e")
    plt.savefig(
        p.folderName + "s_diff_" + str(t).zfill(6) + ".png"
    )  # ,dpi= 100)#_dpi)#,dpi = 100)#, bbox_inches="tight", dpi=100)


def save_s(x, y, s, p, t):
    np.save(p.folderName + "s_" + str(t).zfill(6) + ".npy", operators.get_average(s))


def plot_nu(x, y, s, p, t):
    plt.figure(fig)
    plt.clf()
    nu = 1 - np.mean(np.isnan(s), axis=2).T
    # nut = 1 - np.mean(np.isnan(s), axis=2)
    nu = np.ma.masked_where(nu < p.nu_cs / 7.0, nu)
    # nu[nu > p.nu_cs] = p.nu_cs

    # alpha_vals = 1 - np.exp(-10*nu)

    # print("-------------------------",np.max(nu),np.min(nu),np.max(alpha_vals),np.min(alpha_vals))

    if p.internal_geometry:
        nu = np.ma.masked_where(p.boundary.T, nu)
    plt.pcolormesh(x, y, nu, cmap="inferno_r", shading="auto", vmin=0, vmax=1)
    if p.internal_geometry:
        if p.internal_geometry["perf_plate"]:
            for i in p.internal_geometry["perf_pts"]:
                plt.plot([x[i], x[i]], [y[0], y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if hasattr(p, "plot_colorbar"):
        plt.colorbar(shrink=0.5, location="top", pad=0.01)
        # np.save(p.folderName + "nu_" + str(t).zfill(6) + ".npy",nut)
    plt.savefig(
        p.folderName + "nu_" + str(t).zfill(6) + ".png"
    )  # ,dpi=_dpi)#,dpi= 100)#,dpi =  10000)#,dpi = 120)#,dpi = _dpi)#,dpi = 100)#, bbox_inches="tight", dpi=100)


def save_nu(x, y, s, p, t):
    np.save(p.folderName + "nu_" + str(t).zfill(6) + ".npy", operators.get_solid_fraction(s))


def save_relative_nu(x, y, s, p, t):
    np.save(p.folderName + "nu_" + str(t).zfill(6) + ".npy", operators.get_solid_fraction(s) / p.nu_cs)


def plot_relative_nu(x, y, s, p, t):
    plt.figure(fig)
    nu = 1 - np.mean(np.isnan(s), axis=2).T
    nu /= p.nu_cs
    if p.internal_geometry:
        nu = np.ma.masked_where(p.boundary.T, nu)
    nu = np.ma.masked_where(nu == 0, nu)
    plt.clf()
    plt.pcolormesh(x, y, nu, cmap=bwr, vmin=0, vmax=2)
    if p.internal_geometry:
        if p.internal_geometry["perf_plate"]:
            for i in p.internal_geometry["perf_pts"]:
                plt.plot([x[i], x[i]], [y[0], y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if hasattr(p, "plot_colorbar"):
        plt.colorbar(shrink=0.5, location="top", pad=0.01)  # ,ticks = ticks)
    plt.savefig(p.folderName + "rel_nu_" + str(t).zfill(6) + ".png")  # ,dpi = 120)#,dpi = _dpi)#,dpi = 100)


def plot_u(x, y, s, u, v, p, t):
    plt.figure(fig)
    # mask = mean(isnan(s),axis=2) > 0.95
    # u = ma.masked_where(mask,u/sum(isnan(s),axis=2)).T
    # v = ma.masked_where(mask,v/sum(isnan(s),axis=2)).T
    u = u.T
    v = v.T

    if p.lagrangian:
        u = np.amax(u) - u  # subtract mean horizontal flow
        # v = np.amax(v) - v  # subtract mean horizontal flow

    plt.clf()
    # plt.quiver(X,Y,u,v)
    # print(u)
    plt.pcolormesh(x, y, u, vmin=-np.amax(np.abs(u)), vmax=np.amax(np.abs(u)), cmap="bwr")
    if p.internal_geometry:
        if p.internal_geometry["perf_plate"]:
            for i in p.internal_geometry["perf_pts"]:
                plt.plot([x[i], x[i]], [y[0], y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    if hasattr(p, "plot_colorbar"):
        plt.colorbar(shrink=0.5, location="top", pad=0.01)
    plt.savefig(
        p.folderName + "u_" + str(t).zfill(6) + ".png"
    )  # ,dpi = 120)#,dpi = _dpi)#,dpi = 100)#, bbox_inches="tight", dpi=100)

    plt.clf()
    # plt.quiver(X,Y,u,v)
    plt.pcolormesh(x, y, v, vmin=-np.amax(np.abs(v)), vmax=np.amax(np.abs(v)), cmap="bwr")
    if p.internal_geometry:
        if p.internal_geometry["perf_plate"]:
            for i in p.internal_geometry["perf_pts"]:
                plt.plot([x[i], x[i]], [y[0], y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if hasattr(p, "plot_colorbar"):
        plt.colorbar(shrink=0.5, location="top", pad=0.01)
    plt.savefig(
        p.folderName + "v_" + str(t).zfill(6) + ".png"
    )  # ,dpi = 120)#,dpi = _dpi)#,dpi = 100)#, bbox_inches="tight", dpi=100)

    U = np.sqrt(u**2 + v**2)
    plt.clf()
    plt.pcolormesh(x, y, np.ma.masked_where(p.boundary.T, U), vmin=0, vmax=np.amax(np.abs(U)), cmap=inferno)
    if p.internal_geometry:
        if p.internal_geometry["perf_plate"]:
            for i in p.internal_geometry["perf_pts"]:
                plt.plot([x[i], x[i]], [y[0], y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if hasattr(p, "plot_colorbar"):
        plt.colorbar(shrink=0.5, location="top", pad=0.01)
    plt.savefig(
        p.folderName + "U_mag_" + str(t).zfill(6) + ".png"
    )  # ,dpi = 120)#,dpi = _dpi)#,dpi = 100)#, bbox_inches="tight", dpi=100)


def plot_c(x, y, s, c, p, t):
    # print(np.unique(c))
    plt.figure(fig)

    nm = s.shape[2]
    mask = np.sum(np.isnan(s), axis=2) > 0.95 * nm
    # c = np.ma.masked_where(mask, np.nanmean(c, axis=2))

    plt.clf()
    plt.pcolormesh(
        x, y, np.ma.masked_where(mask, np.nanmean(c, axis=2)).T, cmap="inferno_r", vmin=0, vmax=p.num_cycles
    )
    if p.internal_geometry:
        if p.internal_geometry["perf_plate"]:
            for i in p.internal_geometry["perf_pts"]:
                plt.plot([x[i], x[i]], [y[0], y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if hasattr(p, "plot_colorbar"):
        plt.colorbar(shrink=0.8, location="top", pad=0.01)  # ,ticks = ticks)
        # np.save(p.folderName + "c_" + str(t).zfill(6) + ".npy", np.nanmean(c, axis=2))

    plt.savefig(p.folderName + "c_" + str(t).zfill(6) + ".png")


def save_c(c, folderName, t):
    np.save(folderName + "c_" + str(t).zfill(6) + ".npy", np.nanmean(c, axis=2))


def get_profile(x, y, s, c, p, t):
    # if p.get_ht == True:

    nm = s.shape[2]
    mask = np.sum(np.isnan(s), axis=2) > 0.95 * nm

    creq = np.ma.masked_where(mask, np.nanmean(c, axis=2))

    # den = 1 - np.mean(np.isnan(s), axis=2)
    # den = np.ma.masked_where(den < p.nu_cs / 7.0, den)

    if p.current_cycle == 1:
        val = p.current_cycle
    else:
        val = (p.current_cycle - 1 + p.current_cycle) / 2

    ht = []

    for w in range(p.nx):
        if np.argmin(creq[w]) == 0 and np.ma.is_masked(np.ma.max(creq[w])):
            ht.append(0)

        else:
            # print("RRRRRRRRRRRRRR",creq[w],np.max(creq[w]),np.max(np.nonzero(creq[w] == np.ma.max(creq[w]))))
            ht.append(np.max(np.nonzero(creq[w] == np.max(creq[w]))))
            # ht.append(np.argmax(creq[w]))
            # ht.append(np.argmax(np.nonzero(creq[w] == np.max(creq[w]))))

    np.save(p.folderName + "c_" + str(t).zfill(6) + ".npy", np.nanmean(c, axis=2))
    return ht


def plot_outlet(outlet, folderName):
    plt.figure(summary_fig)

    plt.clf()
    plt.plot(outlet)
    plt.xlabel("time")
    plt.ylabel("outflow")
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    plt.savefig(folderName + "outflow.png")


def plot_profile(x, nu_time_x, p):
    plt.figure(summary_fig)

    plt.clf()
    plt.pcolormesh(x, np.linspace(0, p.t_f, p.nt), nu_time_x)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    plt.savefig(p.folderName + "collapse_profile.png")


def plot_T(x, y, s, T, p, t):
    plt.figure(fig)

    nm = s.shape[2]
    mask = np.sum(np.isnan(s), axis=2) > 0.95 * nm
    plt.clf()
    plt.pcolormesh(
        x,
        y,
        np.ma.masked_where(mask, np.nanmean(T, axis=2)).T,
        cmap=bwr,
        vmin=p.boundary_temperature,
        vmax=p.inlet_temperature,
    )
    plt.axis("off")
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(p.folderName + "T_" + str(t).zfill(6) + ".png")


def make_video(p, p_init):
    if is_ffmpeg_installed:
        fname = p.folderName.split("/")[-2]
        param_study = fname.rsplit("_", 1)[0]
        nice_name = "=".join(fname.rsplit("_", 1))
        subtitle = f"drawtext=text='{nice_name}':x=(w-text_w)/2:y=H-th-10:fontsize=10:fontcolor=white:box=1:boxcolor=black@0.5"
        fps = p.save_inc / p.dt

        for i, video in enumerate(p.videos):
            cmd = [
                "ffmpeg",
                "-y",
                "-r",
                f"{fps}",
                "-pattern_type",
                "glob",
                "-i",
                f"{p.folderName}/{video}_0*.png",
                #  "-c:v", "libx264", "-pix_fmt", "yuv420p"
            ]
            if "nx" in p_init.list_keys or "ny" in p_init.list_keys:
                # add a title to the last video so we know whats going on
                if i == len(p.videos) - 1:
                    # print('a')
                    # cmd.extend(["-vf", '"scale=1000:-1, ' + subtitle + '"']) # make the video 1000 pixels wide
                    cmd.extend(["-vf", "scale=1000:-1:flags=neighbor"])
                else:
                    # print('b')
                    cmd.extend(["-vf", "scale=1000:-1:flags=neighbor"])
            elif i == len(p.videos) - 1:
                # print('c')
                cmd.extend(["-vf", subtitle])
            cmd.extend(["-r", "30", *_video_encoding, f"{p.folderName}/{video}_video.mp4"])
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    else:
        print("ffmpeg not installed, cannot make videos")


def stack_videos(paths, name, videos):
    if is_ffmpeg_installed:
        cmd = [
            "ffmpeg",
            "-y",
        ]

        for video in videos:
            for path in paths:
                cmd.extend(["-i", f"{path}/{video}_video.mp4"])
            pad_string = ""
            for i in range(len(paths)):
                pad_string += f"[{i}]pad=iw+5:color=black[left];[left][{i+1}]"

            cmd.extend(
                [
                    *_video_encoding,
                    # "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-filter_complex",
                    # f"{pad_string}hstack=inputs={len(paths)}",
                    f"hstack=inputs={len(paths)}",
                    f"{video}_videos.mp4",
                ]
            )
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if len(videos) > 1:
            cmd = ["ffmpeg", "-y"]
            for video in videos:
                cmd.extend(["-i", f"{video}_videos.mp4"])
            cmd.extend(
                [
                    *_video_encoding,
                    # "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-filter_complex",
                    f"vstack=inputs={len(videos)}",
                    f"output/{name}.mp4",
                ]
            )
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            cmd = ["rm"]
            for video in videos:
                cmd.append(f"{video}_videos.mp4")
            subprocess.run(cmd)
    else:
        print("ffmpeg not installed, cannot make videos")
