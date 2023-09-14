import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import warnings

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

global fig, summary_fig


def set_plot_size(p):
    global fig, summary_fig
    dpi = 20
    fig = plt.figure(figsize=[p.nx / dpi, p.ny / dpi])
    summary_fig = plt.figure()


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

    plt.clf()
    plt.pcolormesh(np.linspace(0, p.t_f, p.nt), y, s_bar.T, cmap=orange_blue_cmap, vmin=p.s_m, vmax=p.s_M)
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


def plot_s(x, y, s, p, t):
    plt.figure(fig)
    plt.clf()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        s_plot = np.nanmean(s, axis=2).T
    s_plot = np.ma.masked_where(np.isnan(s_plot), s_plot)
    plt.pcolormesh(x, y, s_plot, cmap=orange_blue_cmap, vmin=p.s_m, vmax=p.s_M)
    # plt.colorbar()
    if p.internal_geometry:
        for i in p.internal_geometry["perf_pts"]:
            plt.plot([x[i], x[i]], [y[0], y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(p.folderName + "s_" + str(t).zfill(6) + ".png", dpi=100)


def plot_nu(x, y, s, p, t):
    plt.figure(fig)
    nu = 1 - np.mean(np.isnan(s), axis=2).T
    if p.internal_geometry:
        nu = np.ma.masked_where(p.boundary.T, nu)
    plt.clf()
    plt.pcolormesh(x, y, nu, cmap="inferno", vmin=0, vmax=1)
    if p.internal_geometry:
        if p.internal_geometry["perf_plate"]:
            for i in p.internal_geometry["perf_pts"]:
                plt.plot([x[i], x[i]], [y[0], y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(p.folderName + "nu_" + str(t).zfill(6) + ".png", dpi=100)


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
    plt.savefig(p.folderName + "U_mag_" + str(t).zfill(6) + ".png", dpi=100)


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

def make_video(path, fps=30):
    if is_ffmpeg_installed:
        subprocess.run(["ffmpeg", "-y", "-i", f"{path}/nu_%06d.png", "-r", f"{fps}", f"{path}/nu_video.mp4"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["ffmpeg", "-y", "-i", f"{path}/s_%06d.png",  "-r", f"{fps}", f"{path}/s_video.mp4"],  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print("ffmpeg not installed, cannot make videos")

def stack_videos(paths, name):
    if is_ffmpeg_installed:
        cmd = ["ffmpeg", "-y"]
        for f in paths:
            cmd.extend(["-i", f"{f}/nu_video.mp4"])
        cmd.extend(["-filter_complex",f"hstack=inputs={len(paths)}", "nu_videos.mp4"])
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        cmd = ["ffmpeg", "-y"]
        for f in paths:
            cmd.extend(["-i", f"{f}/s_video.mp4"])
        cmd.extend(["-filter_complex",f"hstack=inputs={len(paths)}", "s_videos.mp4"])
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        cmd = ["ffmpeg","-y", "-i", "nu_videos.mp4", "-i", "s_videos.mp4", "-filter_complex", "vstack=inputs=2", f"{name}.mp4"]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        subprocess.run(['rm', 'nu_videos.mp4', 's_videos.mp4'])
    else:
        print("ffmpeg not installed, cannot make videos")
# ffmpeg -y -i output/collapse/mu_0.5/nu_%06d.png collapse_nu_05.mp4
# ffmpeg -y -i output/collapse/mu_1.0/nu_%06d.png collapse_nu_10.mp4
# ffmpeg -y -i output/collapse/mu_2.0/nu_%06d.png collapse_nu_20.mp4
# ffmpeg -y -i output/collapse/mu_0.5/s_%06d.png collapse_s_05.mp4
# ffmpeg -y -i output/collapse/mu_1.0/s_%06d.png collapse_s_10.mp4
# ffmpeg -y -i output/collapse/mu_2.0/s_%06d.png collapse_s_20.mp4

# ffmpeg -y -i collapse_nu_05.mp4 -i  collapse_nu_10.mp4 -i collapse_nu_20.mp4 -filter_complex vstack=inputs=3 collapse_nu_all.mp4
# rm collapse_nu_05.mp4 collapse_nu_10.mp4 collapse_nu_20.mp4
# ffmpeg -y -i collapse_s_05.mp4 -i  collapse_s_10.mp4 -i collapse_s_20.mp4 -filter_complex vstack=inputs=3 collapse_s_all.mp4
# rm collapse_s_05.mp4 collapse_s_10.mp4 collapse_s_20.mp4
# ffmpeg -y -i collapse_nu_all.mp4 -i collapse_s_all.mp4 -filter_complex hstack=inputs=2 collapse.mp4
# rm collapse_nu_all.mp4 collapse_s_all.mp4

# ffmpeg -y -i output/hopper/mu_0.1/half_width_3/nu_%06d.png hopper_nu_01.mp4
# ffmpeg -y -i output/hopper/mu_1.0/half_width_3/nu_%06d.png hopper_nu_10.mp4
# ffmpeg -y -i output/hopper/mu_10.0/half_width_3/nu_%06d.png hopper_nu_100.mp4
# ffmpeg -y -i output/hopper/mu_0.1/half_width_3/s_%06d.png hopper_s_01.mp4
# ffmpeg -y -i output/hopper/mu_1.0/half_width_3/s_%06d.png hopper_s_10.mp4
# ffmpeg -y -i output/hopper/mu_10.0/half_width_3/s_%06d.png hopper_s_100.mp4