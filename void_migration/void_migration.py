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

# from numba import jit, njit
import params
import plotter
import thermal
import motion
import cycles
import initial


def time_march(p):
    """
    Run the actual simulation(s) as defined in the input json file `p`.
    """

    plotter.set_plot_size(p)

    s_ms = [0, 0]
    change_s_ms = [p.s_m, p.s_m + p.s_m * 0.000001]

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

    if hasattr(p, "charge_discharge"):
        p.nt = cycles.set_nt(p)
    else:
        p.nt = int(np.ceil(p.t_f / p.dt))

    s = initial.IC(p)  # non-dimensional size
    u = np.zeros([p.nx, p.ny])
    v = np.zeros([p.nx, p.ny])

    p_count = np.zeros([p.nt])
    p_count_s = np.zeros([p.nt])
    p_count_l = np.zeros([p.nt])
    non_zero_nu_time = np.zeros([p.nt])

    c = initial.set_concentration(s, X, Y, p)

    initial.set_boundary(s, X, Y, p)

    if hasattr(p, "temperature"):
        T = p.temperature["inlet_temperature"] * np.ones_like(s)
        T[np.isnan(s)] = np.nan
    else:
        T = None

    outlet = []
    plotter.save_coordinate_system(x, y, p)
    plotter.update(x, y, s, u, v, c, T, outlet, p, t, s_ms)

    N_swap = None
    p.indices = np.arange(p.nx * (p.ny - 1) * p.nm)
    np.random.shuffle(p.indices)

    for t in tqdm(range(1, p.nt), leave=False, desc="Time", position=p.concurrent_index + 1):
        outlet.append(0)
        u = np.zeros_like(u)
        v = np.zeros_like(v)

        if hasattr(p, "temperature"):
            T = thermal.update_temperature(s, T, p)

        if hasattr(p, "charge_discharge"):
            p, s_ms = cycles.charge_discharge(p, t, change_s_ms)
            p_count[t], p_count_s[t], p_count_l[t], non_zero_nu_time[t] = cycles.save_quantities(p, s)

        u, v, s, c, T, N_swap = motion.move_voids(u, v, s, p, c=c, T=T, N_swap=N_swap)

        u, v, s, c, outlet = motion.add_voids(u, v, s, p, c, outlet)

        if p.close_voids:
            u, v, s = motion.close_voids(u, v, s)

        if t % p.save_inc == 0:
            plotter.update(x, y, s, u, v, c, T, outlet, p, t, s_ms)

        t += 1
    if hasattr(p, "charge_discharge"):
        plotter.c_d_saves(p, non_zero_nu_time, p_count, p_count_s, p_count_l)
    plotter.update(x, y, s, u, v, c, T, outlet, p, t, s_ms)


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
        if not hasattr(p_init, "max_workers"):
            p_init.max_workers = None

    # run simulations
    all_sims = list(product(*p_init.lists))
    folderNames = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=p_init.max_workers) as executor:
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
