import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import warnings
from operators import get_average, get_solid_fraction

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


def is_ffmpeg_installed():
    try:
        result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0
    except FileNotFoundError:
        return False


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
colors = [(1, 0, 0), (0, 0, 1)]
cmap_name = 'my_list'
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)


global fig, summary_fig


def set_plot_size(p):
    global fig, summary_fig

    # wipe any existing figures
    for i in plt.get_fignums():
        plt.close(i)

    dpi = 20
    fig = plt.figure(figsize=[p.nx / dpi, p.ny / dpi])
    summary_fig = plt.figure()


def update(x, y, s, u, v, c, T, outlet, p, t, *args):
    if "s" in p.plot:
        if hasattr(p,"charge_discharge"):
            plot_s(x, y, s, p, t, *args)
        else:
            plot_s(x, y, s, p, t)
    if "nu" in p.plot:
        plot_nu(x, y, s, p, t)
    if "rel_nu" in p.plot:
        plot_relative_nu(x, y, s, p, t)
    if "U_mag" in p.plot:
        plot_u(x, y, s, u, v, p, t)
    if "concentration" in p.plot:
        plot_c(x, y, s, c, p.folderName, t, p.internal_geometry)
    if "temperature" in p.plot:
        plot_T(x, y, s, T, p, t)
    if "density_profile" in p.plot:
        plot_profile(x, nu_time_x, p)
    if "permeability" in p.plot:
        plot_permeability(x, y, s, p, t)

    if "s" in p.save:
        save_s(x, y, s, p, t)
    if "nu" in p.save:
        save_nu(x, y, s, p, t)
    if "rel_nu" in p.save:
        save_relative_nu(x, y, s, p, t)
    if "U_mag" in p.save:
        save_u(x, y, s, u, v, p, t)
    if "concentration" in p.save:
        save_c(c, p.folderName, t)
    if "outlet" in p.save:
        np.savetxt(p.folderName + "outlet.csv", np.array(outlet), delimiter=",")
    if "temperature" in p.save:
        np.savetxt(p.folderName + "outlet_T.csv", np.array(outlet_T), delimiter=",")
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
    plt.pcolormesh(np.linspace(0, p.t_f, p.nt), y, nu_time.T, cmap="inferno", vmin=0, vmax=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Height (m)")
    plt.colorbar()
    plt.savefig(p.folderName + "nu.png")


def save_coordinate_system(x, y, p):
    np.savetxt(p.folderName + "x.csv", x, delimiter=",")
    np.savetxt(p.folderName + "y.csv", y, delimiter=",")


def c_d_saves(p, non_zero_nu_time, *args):
    np.save(p.folderName + "nu_non_zero_avg.npy", non_zero_nu_time)
    if p.gsd_mode == 'mono':
        np.save(p.folderName + "cell_count.npy", args[0])
    elif p.gsd_mode == 'bi':
        np.save(p.folderName + "cell_count_s.npy", args[0])
        np.save(p.folderName + "cell_count_l.npy", args[1])


def plot_permeability(x, y, s, p, t):
    """
    Calculate and save the permeability of the domain at time t.
    """
    sphericity = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        porosity = np.mean(np.isnan(s), axis=2)
        s_bar = get_average(s)
        permeability = sphericity**2 * (porosity**3) * s_bar**2 / (180 * (1 - porosity) ** 2)

    plt.figure(fig)
    plt.clf()
    plt.pcolormesh(x, y, permeability.T, cmap="inferno")
    plt.axis("off")
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(p.folderName + "permeability_" + str(t).zfill(6) + ".png", dpi=100)


def save_permeability(x, y, s, p, t):
    np.savetxt(p.folderName + "permeability_" + str(t).zfill(6) + ".csv", permeability, delimiter=",")


def plot_s(x, y, s, p, t, *args):
    plt.figure(fig)
    plt.clf()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        s_plot = np.nanmean(s, axis=2).T
    s_plot = np.ma.masked_where(np.isnan(s_plot), s_plot)

    if hasattr(p,"charge_discharge") and p.gsd_mode == 'mono':
        plt.pcolormesh(x, y, s_plot, cmap=cmap, vmin = args[0][0], vmax = args[0][1])
    else:
        plt.pcolormesh(x, y, s_plot, cmap=orange_blue_cmap, vmin=p.s_m, vmax=p.s_M)
        # plt.colorbar()

    if p.internal_geometry:
        for i in p.internal_geometry["perf_pts"]:
            plt.plot([x[i], x[i]], [y[0], y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    ticks = np.linspace(p.s_m, p.s_M, 3, endpoint=True)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if hasattr(p,"plot_colorbar"):
        plt.colorbar(shrink=0.8,location='top',pad = 0.01,ticks = ticks)
    plt.savefig(p.folderName + "s_" + str(t).zfill(6) + ".png")


def save_s(x, y, s, p, t):
    np.save(p.folderName + "s_" + str(t).zfill(6) + ".npy", get_average(s))


def plot_nu(x, y, s, p, t):
    plt.figure(fig)
    nu = 1 - np.mean(np.isnan(s), axis=2).T
    if p.internal_geometry:
        nu = np.ma.masked_where(p.boundary.T, nu)
    plt.clf()
    plt.pcolormesh(x, y, nu, cmap="inferno_r", vmin=0, vmax=1)
    if p.internal_geometry:
        if p.internal_geometry["perf_plate"]:
            for i in p.internal_geometry["perf_pts"]:
                plt.plot([x[i], x[i]], [y[0], y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if hasattr(p,"plot_colorbar"):
        plt.colorbar(shrink=0.8,location='top',pad = 0.01)
    plt.savefig(p.folderName + "nu_" + str(t).zfill(6) + ".png")


def save_nu(x, y, s, p, t):
    np.save(p.folderName + "nu_" + str(t).zfill(6) + ".npy", get_solid_fraction(s))


def save_relative_nu(x, y, s, p, t):
    np.save(p.folderName + "nu_" + str(t).zfill(6) + ".npy", get_solid_fraction(s) / p.nu_cs)


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
    if hasattr(p,"plot_colorbar"):
        plt.colorbar(shrink=0.8,location='top',pad = 0.01)#,ticks = ticks)
    plt.savefig(p.folderName + "rel_nu_" + str(t).zfill(6) + ".png")


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
    # plt.colorbar()
    plt.savefig(p.folderName + "u_" + str(t).zfill(6) + ".png", dpi=100)

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
    plt.savefig(p.folderName + "v_" + str(t).zfill(6) + ".png", dpi=100)

    U = np.sqrt(u**2 + v**2)
    plt.clf()
    plt.pcolormesh(x, y, np.ma.masked_where(p.boundary.T, U), vmin=0, vmax=np.amax(np.abs(U)), cmap="inferno")
    if p.internal_geometry:
        if p.internal_geometry["perf_plate"]:
            for i in p.internal_geometry["perf_pts"]:
                plt.plot([x[i], x[i]], [y[0], y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if hasattr(p,"plot_colorbar"):
        plt.colorbar(shrink=0.8, location='top', pad = 0.01)#,ticks = ticks)
    plt.savefig(p.folderName + "U_mag_" + str(t).zfill(6) + ".png")


def plot_c(x, y, s, c, folderName, t, internal_geometry):
    plt.figure(fig)

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


def save_c(c, folderName, t):
    np.save(folderName + "c_" + str(t).zfill(6) + ".npy", np.nanmean(c, axis=2))


def plot_outlet(outlet, folderName):
    plt.figure(summary_fig)

    plt.clf()
    plt.plot(outlet)
    plt.xlabel("time")
    plt.ylabel("outflow")
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    plt.savefig(folderName + "outflow.png", dpi=100)


def plot_profile(x, nu_time_x, p):
    plt.figure(summary_fig)

    plt.clf()
    plt.pcolormesh(x, np.linspace(0, p.t_f, p.nt), nu_time_x)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    plt.savefig(p.folderName + "collapse_profile.png", dpi=100)


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
    plt.savefig(p.folderName + "T_" + str(t).zfill(6) + ".png", dpi=100)


def make_video(p):
    if is_ffmpeg_installed:
        fname = p.folderName.split("/")[-2]
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
            # add a title to the last video so we know whats going on
            if i == len(p.videos) - 1:
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
