#!/usr/bin/python

__doc__ = """
void_migration.py

This script simulates the migration of voids in a granular material.
"""
__author__ = "Benjy Marks, Shivakumar Athani"
__version__ = "0.3"

import sys
import numpy as np
import concurrent.futures
from tqdm.auto import tqdm
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

    p = params.update_before_time_march(p, cycles)

    s = initial.IC(p)  # non-dimensional size
    u = np.zeros([p.nx, p.ny])
    v = np.zeros([p.nx, p.ny])
    p_count = np.zeros([p.nt])
    p_count_s = np.zeros([p.nt])
    p_count_l = np.zeros([p.nt])
    non_zero_nu_time = np.zeros([p.nt])

    c = initial.set_concentration(s, p.X, p.Y, p)

    initial.set_boundary(s, p.X, p.Y, p)

    if hasattr(p, "temperature"):
        T = p.temperature["inlet_temperature"] * np.ones_like(s)
        T[np.isnan(s)] = np.nan
    else:
        T = None

    outlet = []
    if len(p.save) > 0:
        plotter.save_coordinate_system(p.x, p.y, p)
    plotter.update(p.x, p.y, s, u, v, c, T, outlet, p, 0)

    N_swap = None
    p.indices = np.arange(p.nx * (p.ny - 1) * p.nm)
    np.random.shuffle(p.indices)

    for t in tqdm(range(1, p.nt), leave=False, desc="Time", position=p.concurrent_index + 1):
        outlet.append(0)
        u = np.zeros_like(u)
        v = np.zeros_like(v)

        if hasattr(p, "temperature"):
            T = thermal.update_temperature(s, T, p)

        if p.charge_discharge:
            p = cycles.charge_discharge(p, t)
            p_count[t], p_count_s[t], p_count_l[t], non_zero_nu_time[t] = cycles.save_quantities(p, s)

        if p.vectorized:
            u, v, s, c, T, N_swap = motion.move_voids_fast(u, v, s, p, c=c, T=T, N_swap=N_swap)
        else:
            u, v, s, c, T, N_swap = motion.move_voids(u, v, s, p, c=c, T=T, N_swap=N_swap)

        u, v, s, c, outlet = motion.add_voids(u, v, s, p, c, outlet)

        if p.close_voids:
            u, v, s = motion.close_voids(u, v, s)

        if t % p.save_inc == 0:
            plotter.update(p.x, p.y, s, u, v, c, T, outlet, p, t)

    plotter.update(p.x, p.y, s, u, v, c, T, outlet, p, t)


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
