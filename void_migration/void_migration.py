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
import time
import plotter
import thermal
import motion
import cycles
import initial
import cProfile


def time_march(p):
    """
    Run the actual simulation(s) as defined in the input json file `p`.
    """

    plotter.set_plot_size(p)

    y = np.linspace(0, p.H, p.ny)
    p.dy = y[1] - y[0]

    # x = np.linspace(-p.nx * p.dy / 2, p.nx * p.dy / 2, p.nx)  # force equal grid spacing
    if p.silo_width == "half":
        x = np.linspace(0, 22, p.nx)
    elif p.silo_width == "full":
        x = np.linspace(-22, 22, p.nx)
    p.dx = x[1] - x[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    y += p.dy / 2.0
    t = 0

    # plotter.set_plot_size(p)

    # y = np.linspace(0, p.H, p.ny)
    # p.dy = y[1] - y[0]
    # x = np.linspace(-p.nx * p.dy / 2, p.nx * p.dy / 2, p.nx)  # force equal grid spacing
    # X, Y = np.meshgrid(x, y, indexing="ij")

    # y += p.dy / 2.0
    # t = 0

    p.t_p = p.s_m / np.sqrt(p.g * p.H)  # smallest confinement timescale (at bottom) (s)
    p.free_fall_velocity = np.sqrt(p.g * (p.s_m + p.s_M) / 2.0)  # time to fall one mean diameter (s)
    surface_profile = []
    safe = False
    stability = 0.5
    while not safe:
        p.P_u_ref = stability
        p.dt = p.P_u_ref * p.dy / p.free_fall_velocity

        p.P_lr_ref = p.alpha * p.P_u_ref

        if p.gsd_mode == "fbi":
            p.P_u_max = p.P_u_ref * (p.Fr * p.s_M / p.s_m)
            p.P_lr_max = p.P_lr_ref * (p.Fr * p.s_M / p.s_m)

        else:
            p.P_u_max = p.P_u_ref * (p.s_M / p.s_m)
            p.P_lr_max = p.P_lr_ref * (p.s_M / p.s_m)

        if p.P_u_max + 2 * p.P_lr_max <= 1:
            safe = True
        else:
            stability *= 0.95

    # print("********************",p.P_u_max,p.P_lr_max)

    if hasattr(p, "charge_discharge"):
        p.nt = cycles.set_nt(p)
    else:
        p.nt = int(np.ceil(p.t_f / p.dt))

    s = initial.IC(p)  # non-dimensional size
    p.tmp_s = np.zeros_like(s)
    p.tmp_s = s.copy()
    s = initial.inclination(p, s)  # masks the region depending upon slope angle
    u = np.zeros([p.nx, p.ny])
    v = np.zeros([p.nx, p.ny])
    p_count = np.zeros([p.nt])
    p_count_s = np.zeros([p.nt])
    p_count_l = np.zeros([p.nt])
    non_zero_nu_time = np.zeros([p.nt])

    c = initial.set_concentration(s, X, Y, p)

    initial.set_boundary(s, X, Y, p)

    ### points to store infor about grainsize distribution ###
    if p.silo_width == "half":
        xpoints = np.arange(int(0.1 * p.nx), int(p.nx), int(0.1 * p.nx))
        ypoints = np.arange(int(0.1 * p.ny), int(p.ny), int(0.1 * p.ny))

    if p.silo_width == "full":
        xpoints = np.arange(int(0.1 * p.nx), int(p.nx / 2), int(0.1 * p.nx))
        ypoints = np.arange(int(0.1 * p.ny), int(p.ny), int(0.1 * p.ny))
    ##########################################################

    if hasattr(p, "temperature"):
        T = p.temperature["inlet_temperature"] * np.ones_like(s)
        T[np.isnan(s)] = np.nan
    else:
        T = None

    outlet = []
    plotter.save_coordinate_system(x, y, p)
    plotter.update(x, y, s, u, v, c, T, outlet, p, t)

    N_swap = None
    # p.indices = np.arange(p.nx * (p.ny - 1) * p.nm)
    # np.random.shuffle(p.indices)

    for t in tqdm(range(1, p.nt), leave=False, desc="Time", position=p.concurrent_index + 1):
        outlet.append(0)
        u = np.zeros_like(u)
        v = np.zeros_like(v)

        if hasattr(p, "temperature"):
            T = thermal.update_temperature(s, T, p)

        if hasattr(p, "charge_discharge"):
            Mass_inside = np.count_nonzero(~np.isnan(s)) * p.M_of_each_cell
            # print("CCCCCCCCCCCCCCCCCCCCCCC",Mass_inside,np.count_nonzero(~np.isnan(s)))
            p = cycles.charge_discharge(p, t, Mass_inside)
            p_count[t], p_count_s[t], p_count_l[t], non_zero_nu_time[t] = cycles.save_quantities(p, s)
            if p.get_ht == True:
                ht = plotter.get_profile(x, y, s, c, p, t)
                surface_profile.append(ht)
        # start_t = time.time()
        u, v, s, c, T, N_swap = motion.move_voids(u, v, s, p, c=c, T=T, N_swap=N_swap)
        # end_t = time.time()
        # print("MMMMMMMMMMMMMMM", end_t - start_t)

        # start_t = time.time()
        u, v, s, c, outlet = motion.add_voids(u, v, s, p, c, outlet)
        # end_t = time.time()
        # print("AAAAAAAAAAAAAAA", end_t - start_t)

        if p.close_voids:
            u, v, s = motion.close_voids(u, v, s)

        if t % p.save_inc == 0:
            plotter.update(x, y, s, u, v, c, T, outlet, p, t)
            # if hasattr(p, "charge_discharge") and (p.gsd_mode == 'bi' or p.gsd_mode == 'fbi'):
            #     plotter.plot_pdf_cdf(p,s,xpoints,ypoints,t)

            p.tmp_s = s.copy()
        t += 1
    if hasattr(p, "charge_discharge"):
        plotter.c_d_saves(p, non_zero_nu_time, p_count, p_count_s, p_count_l)
    plotter.update(x, y, s, u, v, c, T, outlet, p, t)

    np.save(p.folderName + "surface_profiles.npy", surface_profile)


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
    # plotter.make_video(p, p_init)
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

    # if len(all_sims) > 1:
    #     plotter.stack_videos(folderNames, dict["input_filename"], p_init.videos)
