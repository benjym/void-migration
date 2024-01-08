import numpy as np

op_arr = []
s_ms = [0, 0]


def charge_discharge(p, t, change_s_ms):
    """
    As of now two times (t_fill and t_empty) are calculated and t_settle is given
    t_fill - filling time
    t_settle - allow the cells to settle
    t_empty - end time
    for mono-disperse conditions, increase very slightly the grainsize for different cycles
    (mainly to assign different colors for different cycle), and
    this is done in the time_march function
    """

    # if p.gsd_mode == "mono":
    #     colsm = []
    #     for i in range(p.no_of_cycles):
    #         if i % 2 == 0:
    #             colsm.append(change_s_ms[0])
    #         else:
    #             colsm.append(change_s_ms[1])

    res = [sub["t_empty"] for sub in op_arr]  # Just take the t_empty values from the op_arr dictionary

    yj = []
    for j in range(len(res)):
        if t < int(np.ceil(res[j] / p.dt)):
            yj.append(op_arr[j])

    # if p.gsd_mode == "mono":
    #     col_index = colsm[int(len(op_arr) - len(yj))]

    if t <= int(
        np.ceil(yj[0].get("t_fill") / p.dt)
    ):  # To always pick the time belonging to the first element in the array
        if p.gsd_mode == "mono":
            # p.s_m = col_index
            p.s_M = p.s_m
            p.add_voids = "place_on_top"  # 'pour_base'
        else:
            p.add_voids = "place_on_top"

    elif int(np.ceil(yj[0].get("t_fill") / p.dt)) < t <= int(np.ceil(yj[0].get("t_settle") / p.dt)):
        p.add_voids = "None"

    elif int(np.ceil(yj[0].get("t_settle") / p.dt)) < t < int(np.ceil(yj[0].get("t_empty") / p.dt) - 1):
        p.add_voids = "central_outlet"
        p.save_outlet = True

    p.current_cycle = len(op_arr) - len(yj) + 1
    p.num_cycles = len(op_arr)

    return p, change_s_ms


def cals_in_cd(p):
    """
    Calculates the time required for filling, settling and emptying for each cycle and a desired number of cycles
    M_total : Total mass for a given solid fraction
    Mass_in_per_u_t : Mass going in per unit time
    Mass_out_per_u_t : Mass coming out of silo per unit time
    T_in : filling time calculated based on M_total, Mass_in_per_u_t and p.dt
    T_settle : settling time
    T_out : emptying time calculated based on M_in, Mass_out_per_u_t and p.dt
    op_arr : holds the values for t_fill, t_settle and t_empty for each cycle and for a desired number of cycles

    """
    if hasattr(p, "give_mass"):
        M_of_each_cell = p.dy * p.dy * p.rho_p / p.nm

        # Total mass = no. of non-nan cells * Mass of each cell
        M_total = p.nx * p.ny * p.nm * M_of_each_cell * p.nu_cs

        # Mass per unit time
        Mass_in_per_u_t = (p.half_width * 2 + 1) * p.nm * p.fill_ratio * M_of_each_cell
        Mass_out_per_u_t = (p.half_width * 2 + 1) * p.nm * p.free_par * M_of_each_cell

        # Mass in
        Mass_in = M_total * np.array(list(p.M_ins.values()))

        # Mass out
        Mass_out = Mass_in * np.array(list(p.M_empty.values()))

        # Time in
        T_in = [
            int((Mass_in[0] / Mass_in_per_u_t) * p.dt),
            int((Mass_in[1] / Mass_in_per_u_t) * p.dt),
            int((Mass_in[2] / Mass_in_per_u_t) * p.dt),
        ]

        # Time settle
        T_settle = np.array(list(p.T_settle.values()))

        # Time out
        T_out = [
            int((Mass_out[0] / Mass_out_per_u_t) * p.dt),
            int((Mass_out[1] / Mass_out_per_u_t) * p.dt),
            int((Mass_out[2] / Mass_out_per_u_t) * p.dt),
        ]

        tmp = 0

        for i in range(p.no_of_cycles):
            T_in[i] = T_in[i] + tmp
            T_settle[i] = T_settle[i] + T_in[i]
            T_out[i] = T_out[i] + T_settle[i]
            tmp = T_out[i]

    if hasattr(p, "give_time") and hasattr(p, "T_in") and hasattr(p, "T_settle") and hasattr(p, "T_empty"):
        T_in = list(p.T_in.values())
        T_settle = list(p.T_settle.values())
        T_out = list(p.T_empty.values())
        tmp = 0
        for i in range(p.no_of_cycles):
            T_in[i] = T_in[i] + tmp
            T_settle[i] = T_settle[i] + T_in[i]
            T_out[i] = T_out[i] + T_settle[i]
            tmp = T_out[i]

    all_Ts = np.transpose([T_in, T_settle, T_out])  # to arange the times in a cycle fashion

    # Initialize dictionary
    times = ["t_fill", "t_settle", "t_empty"]

    for i in range(p.no_of_cycles):
        op_arr.append({x: {} for x in times})  # creating empty dictionary with empty times

    for j in range(p.no_of_cycles):
        op_arr[j] = {
            key: all_Ts[j][i] for i, key in enumerate(op_arr[j])
        }  # assigned the values for times t_fill t_settle and t_empty

    return op_arr


def save_quantities(p, s):
    p_count = 0
    p_count_s = 0
    p_count_l = 0
    non_zero_nu_time = 0

    if p.gsd_mode == "mono":
        p_count = np.count_nonzero(s[~np.isnan(s)])
    elif p.gsd_mode == "bi":
        p_count_s = np.count_nonzero(s[~np.isnan(s)] == p.s_m)
        p_count_l = np.count_nonzero(s[~np.isnan(s)] == p.s_M)

    non_zero_avg_nu = 1 - np.mean(np.isnan(s), axis=2)  # get the solid fraction
    non_zero_nu_time = np.mean(
        non_zero_avg_nu[np.nonzero(non_zero_avg_nu)]
    )  # get the avg solid fraction ignoring the zeros

    return p_count, p_count_s, p_count_l, non_zero_nu_time


def set_nt(p):
    if (hasattr(p, "give_mass")) or (
        hasattr(p, "give_time") and hasattr(p, "T_in") and hasattr(p, "T_settle") and hasattr(p, "T_empty")
    ):
        op_arr = cals_in_cd(p)  # get the time cycles

    else:
        op_arr = p.T_cycles

    return int(np.ceil(op_arr[-1].get("t_empty") / p.dt))  # cal the number of steps based on final t_empty
