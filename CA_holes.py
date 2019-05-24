#!/usr/bin/python

__doc__ = """
CA_holes.py

This script simulates ....
"""
__author__ = "benjy"
__version__ = "2.2 31/10/2012"

import os, sys
from numpy import *
import warnings
warnings.filterwarnings("ignore")
from plotter import *

def IC():
    if mode == 'mono': s = ones([nx,ny,nz])
    if mode == 'bi':
        s = random.choice([s_m,1],size=[nx,ny,nm])
    elif mode == 'poly':
        s_0 = s_m/(1.-s_m) # intermediate calculation
        s = random.rand(nx,ny,nm)
        s = (s + s_0)/(s_0 + 1.) # now between s_m and 1

    if IC_mode == 'random':
        mask = random.rand(nx,ny,nm)<fill
    elif IC_mode == 'top':
        mask = zeros([nx,ny,nm],dtype=bool)
        mask[:,int(top_fill*ny):,:] = True
    elif IC_mode == 'full':
        mask = zeros_like(s,dtype=bool)
    elif IC_mode == 'slope':
        mask = zeros_like(s,dtype=bool)
        mask[:,int(0.7*ny):,:] = True
        s[Y<(-0.5*X + 0.5)] = -1
    elif IC_mode == 'column':
        # mask = zeros_like(s,dtype=bool)
        mask = random.rand(nx,ny,nm)<fill
        # mask[:nx//3:,:,:] = True
        # mask[2*nx//3:,:,:] = True
        mask[:nx//2-int(ny*aspect_ratio/2.),:,:] = True
        mask[nx//2+int(ny*aspect_ratio/2.):,:,:] = True
        # mask[nx//2:nx//2+int(ny*aspect_ratio),:,:] = True
    elif IC_mode == 'empty':
        mask = ones([nx,ny,nm],dtype=bool)
    s[mask] = nan
    return s

def move_holes_adv(u,v,s): # Advection
    for j in range(ny-2,-1,-1):
        x_loop = arange(nx)
        if internal_geometry: x_loop = x_loop[~boundary[:,j]] # don't move apparent voids at boundaries
        random.shuffle(x_loop)
        for i in x_loop:
            for k in range(nm):
                if isnan(s[i,j,k]):
                    if random.rand() < P_adv:
                        if not isnan(s[i,j+1,k]):
                            if internal_geometry and not boundary[i,j+1]:
                                s[i,j,k], s[i,j+1,k] = s[i,j+1,k], s[i,j,k]
                                v[i,j] += 1
                                if temp_mode == 'temperature': T[i,j,k], T[i,j+1,k] = T[i,j+1,k], T[i,j,k]
    return u,v,s

def move_holes_diff(u,v,s): # Diffusion
    for j in range(ny-2,-1,-1):
        x_loop = arange(nx)
        if internal_geometry: x_loop = x_loop[~boundary[:,j]] # don't move apparent voids at boundaries
        random.shuffle(x_loop)
        for i in x_loop:
            for k in range(nm):
                if isnan(s[i,j,k]):
                    if random.rand() < P_diff:
                        # CAN MAKE SEG BY SETTING PROBABILITY OF lr TO DEPEND ON SIZE RATIO
                        # lr = random.rand() < 1./(1. + s[i-1,j,k]/s[i+1,j,k] ) # just segregation
                        # lr = random.rand() < 0.5*(1 - sin(radians(theta))) # just slope
                        if i == 0:
                            if cyclic_BC:
                                lr =  random.rand() < (1 - sin(radians(theta)))/(1 - sin(radians(theta)) + (1+sin(radians(theta)))*(s[-1,j,k]/s[1,j,k]) ) # both together
                                lr = 2*lr - 1 # rescale to +/- 1
                            else:
                                lr = 1
                        elif i == nx - 1:
                            if cyclic_BC:
                                lr =  random.rand() < (1 - sin(radians(theta)))/(1 - sin(radians(theta)) + (1+sin(radians(theta)))*(s[-2,j,k]/s[0,j,k]) ) # both together
                                lr = 2*lr - 1 # rescale to +/- 1
                            else:
                                lr = -1
                        else:
                            if boundary[i-1,j]: l = 1.0001e10 # zero chance of moving there
                            else: l = s[i-1,j,k]
                            if boundary[i+1,j]: r = 1e10 # zero chance of moving there
                            else: r = s[i+1,j,k]
                            lr =  random.rand() < (1 - sin(radians(theta)))/(1 - sin(radians(theta)) + (1+sin(radians(theta)))*(l/r) ) # both together
                            print(s[i-1,j,k],s[i,j,k],s[i+1,j,k],(1 - sin(radians(theta)))/(1 - sin(radians(theta)) + (1+sin(radians(theta)))*(l/r) ))
                            lr = 2*lr - 1 # rescale to +/- 1

                        if i == nx-1 and lr == 1: # right boundary
                            if cyclic_BC:
                                # if not isnan(s[0,j+1,k]): # this sets the angle of repose?
                                if not isnan(s[0,j,k]):
                                    s[-1,j,k], s[0,j,k] = s[0,j,k], s[-1,j,k]
                                    u[i,j] -= lr
                                    if temp_mode == 'temperature': T[-1,j,k], T[0,j,k] = T[0,j,k], T[-1,j,k]
                        elif i == 0 and lr == -1: # left boundary
                            if cyclic_BC:
                                # if not isnan(s[i+lr,j+1,k]): # this sets the angle of repose?
                                if not isnan(s[-1,j,k]):
                                    s[0,j,k], s[-1,j,k] = s[-1,j,k], s[0,j,k]
                                    u[i,j] -= lr
                                    if temp_mode == 'temperature': T[0,j,k], T[-1,j,k] = T[-1,j,k], T[0,j,k]
                        else:
                            if not isnan(s[i+lr,j,k]):
                            # if not isnan(s[i+lr,j+1,k]): # this sets the angle of repose at 45
                            # if mean(isnan(s[i,j,:]))>0.5: # if here is mostly empty (ie outside mass)
                            #     if mu < 1: A = mu/2. # proportion of cell that should be filled diagonally up
                            #     else: A = 1. - 1./(2.*mu)
                            #     if mean(~isnan(s[i+lr,j+1,:]))>A: # this sets an angle of repose?
                            #         s[i,j,k], s[i+lr,j,k] = s[i+lr,j,k], s[i,j,k]
                            #         u[i,j] -= lr
                            # else:
                                if internal_geometry and not boundary[i+lr,j]:
                                    s[i,j,k], s[i+lr,j,k] = s[i+lr,j,k], s[i,j,k]
                                    u[i,j] -= lr
                                    if temp_mode == 'temperature': T[i,j,k], T[i+lr,j,k] = T[i+lr,j,k], T[i,j,k]

    return u,v,s

def move_holes(u,v,s):
    for j in range(ny-2,-1,-1):
        if internal_geometry: x_loop = arange(nx)[~boundary[:,j]] # don't move apparent voids at boundaries
        else: x_loop = arange(nx)
        random.shuffle(x_loop)
        for i in x_loop:
            for k in range(nm):
                if isnan(s[i,j,k]):
                    # t_p = dy/sqrt(g*(H-y[j])) # local confinement timescale (s)

                    # if random.rand() < free_fall_velocity*dt/dy:
                        # up
                        if isnan(s[i,j+1,k]): P_u = 0
                        # else: P_u = 1./s[i,j+1,k]/(2.*mu)
                        else: P_u = 2./(2.*mu)/s[i,j+1,k] # FIXME ????

                        # left
                        if i > 0:
                            # if ( not isnan(s[i-1,j,k]) and not isnan(s[i-1,j+1,k]) ): # LEFT
                                # P_l = (0.5 + 0.5*sin(radians(theta)))/s[i-1,j,k]
                            if not isnan(s[i-1,j+1,k]): # UP LEFT
                                P_l = (0.5 + 0.5*sin(radians(theta)))/s[i-1,j+1,k]
                            else:
                                P_l = 0 # P_r + P_l = 1 at s=1

                            if internal_geometry:
                                if boundary[i-1,j+1]: P_l *= perf_rate
                            # if perf_plate and i-1==perf_pts[0]: P_l *= perf_rate
                            # if perf_plate and i-1==perf_pts[1]: P_l *= perf_rate
                        elif cyclic_BC:
                            # if isnan(s[-1,j,k]): P_l = 0 # LEFT
                            # else: P_l = (0.5 + 0.5*sin(radians(theta)))/s[-1,j,k]
                            if isnan(s[-1,j+1,k]): P_l = 0 # UP LEFT
                            else: P_l = (0.5 + 0.5*sin(radians(theta)))/s[-1,j+1,k]
                        else:
                            P_l = 0

                        # right
                        if i+1 < nx:
                            # if ( not isnan(s[i+1,j,k]) and not isnan(s[i+1,j+1,k]) ): # RIGHT
                                # P_r = (0.5 - 0.5*sin(radians(theta)))/s[i+1,j,k]
                            if not isnan(s[i+1,j+1,k]): # UP RIGHT
                                P_r = (0.5 - 0.5*sin(radians(theta)))/s[i+1,j+1,k]
                            else:
                                P_r = 0

                            if internal_geometry:
                                if boundary[i+1,j+1]: P_r *= perf_rate
                            # if perf_plate and i+1==perf_pts[0]: P_r *= perf_rate
                            # if perf_plate and i+1==perf_pts[1]: P_r *= perf_rate
                        elif cyclic_BC:
                            # if isnan(s[0,j,k]): P_r = 0 # RIGHT
                            # else: P_r = (0.5 - 0.5*sin(radians(theta)))/s[0,j,k]
                            if isnan(s[0,j+1,k]): P_r = 0 # UP RIGHT
                            else: P_r = (0.5 - 0.5*sin(radians(theta)))/s[0,j+1,k]
                        else:
                            P_r = 0

                        P_tot = P_u + P_l + P_r

                        if P_tot > 0:
                            P = random.rand()
                            if P < P_u/P_tot and P_u > 0: # go up
                                s[i,j,k], s[i,j+1,k] = s[i,j+1,k], s[i,j,k]
                                if temp_mode == 'WGCP': c[i,j,k], c[i,j+1,k] = c[i,j+1,k], c[i,j,k]
                                if temp_mode == 'temperature': T[i,j,k], T[i,j+1,k] = T[i,j+1,k], T[i,j,k]
                                v[i,j] += 1
                            elif P < (P_l + P_u)/P_tot: # go left
                                # s[i,j,k], s[i-1,j,k] = s[i-1,j,k], s[i,j,k] # LEFT
                                # if temp_mode == 'WGCP': c[i,j,k], c[i-1,j,k] = c[i-1,j,k], c[i,j,k]
                                s[i,j,k], s[i-1,j+1,k] = s[i-1,j+1,k], s[i,j,k] # UP LEFT
                                if temp_mode == 'WGCP': c[i,j,k], c[i-1,j+1,k] = c[i-1,j+1,k], c[i,j,k]
                                if temp_mode == 'temperature': T[i,j,k], T[i-1,j+1,k] = T[i-1,j+1,k], T[i,j,k]
                                # u[i,j] += 1 # LEFT
                                # v[i,j] += 1
                                u[i,j] += sqrt(2) # UP LEFT
                                v[i,j] += sqrt(2)
                            elif P < (P_l + P_u + P_r)/P_tot: # go right
                                if i+1 < nx:
                                    # s[i,j,k], s[i+1,j,k] = s[i+1,j,k], s[i,j,k] # RIGHT
                                    # if temp_mode == 'WGCP': c[i,j,k], c[i+1,j,k] = c[i+1,j,k], c[i,j,k]
                                    s[i,j,k], s[i+1,j+1,k] = s[i+1,j+1,k], s[i,j,k] # UP RIGHT
                                    if temp_mode == 'WGCP': c[i,j,k], c[i+1,j+1,k] = c[i+1,j+1,k], c[i,j,k]
                                    if temp_mode == 'temperature': T[i,j,k], T[i+1,j+1,k] = T[i+1,j+1,k], T[i,j,k]
                                else:
                                    # s[i,j,k], s[0,j,k] = s[0,j,k], s[i,j,k] # RIGHT
                                    s[i,j,k], s[0,j+1,k] = s[0,j+1,k], s[i,j,k] # UP RIGHT
                                    if temp_mode == 'WGCP': c[i,j,k], c[0,j+1,k] = c[0,j+1,k], c[i,j,k]
                                    if temp_mode == 'temperature': T[i,j,k], T[0,j+1,k] = T[0,j+1,k], T[i,j,k]
                                # u[i,j] -= 1 # RIGHT
                                # v[i,j] += 1
                                u[i,j] -= sqrt(2) # UP RIGHT
                                v[i,j] += sqrt(2)
    return u,v,s

def add_temp(u,v,s):
    if temp_mode == 'hopper': # Remove at central outlet
        for i in range(nx//2-half_width,nx//2+half_width+1):
            for k in range(nm):
                # if random.rand() < Tg:
                    if not isnan(s[i,0,k]):
                        if refill:
                            if sum(isnan(s[nx//2-half_width:nx//2+half_width+1,-1,k])) > 0:
                                target = random.choice(nonzero(isnan(s[:,-1,k]))[0])
                                s[target,-1,k], s[i,0,k] = s[i,0,k], s[target,-1,k]
                        else:
                            s[i,0,k] = nan
                        outlet[-1] += 1
    elif temp_mode == 'temperature': # Remove at central outlet
        for i in range(nx//2-half_width,nx//2+half_width+1):
            for k in range(nm):
                # if random.rand() < Tg:
                    if not isnan(s[i,0,k]):
                        if refill:
                            if sum(isnan(s[nx//2-half_width:nx//2+half_width+1,-1,k])) > 0:
                                if internal_geometry: target = nx//2 - half_width + random.choice(nonzero(isnan(s[nx//2-half_width:nx//2+half_width+1,-1,k]))[0]) # HACK
                                else: target = random.choice(nonzero(isnan(s[:,-1,k]))[0])
                                s[target,-1,k], s[i,0,k] = s[i,0,k], s[target,-1,k]
                                T[target,-1,k] = inlet_temperature
                                outlet_T.append(T[i,0,k])
                        else:
                            s[i,0,k] = nan
                        outlet[-1] += 1
    elif temp_mode == 'WGCP': # Remove at multiple points in base
        for l,source_pt in enumerate(source_pts):
            for i in range(source_pt-half_width,source_pt+half_width+1):
                for k in range(nm):
                    if random.rand() < Tg[l]:
                        target = random.choice(nonzero(isnan(s[:,-1,k]))[0])
                        s[target,-1,k] = s[i,0,k]
                        if target <= perf_pts[0]: c[target,-1,k] = 0
                        elif target <= perf_pts[1]: c[target,-1,k] = 1
                        else: c[target,-1,k] = 2
                        s[i,0,k] = nan
    elif temp_mode == 'slope': # Add voids at base
        for i in range(nx):
            for k in range(nm):
                if not isnan(s[i,0,k]):
                    # MOVE UP TO FIRST VOID
                    # if ( random.rand() < (free_fall_velocity*dt)/(Tg*H) and sum(isnan(s[i,:,k])) ) > 0: # Tg is relative height (out of the maximum depth) that voids should rise to before being filled
                        # first_void = isnan(s[i,:,k]).nonzero()[0][0]
                        # v[i,:first_void+1] += isnan(s[i,:first_void+1,k])
                        # s[i,:first_void+1,k] = roll(s[i,:first_void+1,k],1)
                    # MOVE EVERYTHING UP
                    if ( random.rand() < Tg*dt/dy and sum(isnan(s[i,:,k])) ) > 0:
                        if isnan(s[i,-1,k]):
                            v[i,:] += 1 #isnan(s[i,:,k])
                            s[i,:,k] = roll(s[i,:,k],1)
    elif temp_mode == 'mara': # Add voids at base
        # for i in range(5,nx-5):
        for i in range(nx):
            for k in range(nm):
                if not isnan(s[i,0,k]):
                    if random.rand() < Tg*dt/dy and sum(isnan(s[i,:,k])) > 0:
                        first_void = isnan(s[i,:,k]).nonzero()[0][0]
                        v[i,:first_void+1] += isnan(s[i,:first_void+1,k])
                        s[i,:first_void+1,k] = roll(s[i,:first_void+1,k],1)
    elif temp_mode == 'diff_test': # Remove at outlet
        if t == 0: s[nx//2,0,:] = nan
    elif temp_mode == 'pour': # pour in centre at top
        s[nx//2-half_width:nx//2+half_width+1,-1,:] = 1.
    return u,v,s

def generate(u,v,s): # Moving voids create voids
    U = sqrt(u**2 + v**2)
    for i in range(nx):
        for j in range(ny):
            for k in range(nm):
                if not isnan(s[i,j,k]):
                    if random.rand() < 1*U[i,j]/nm*dt/dy: # FIXME
                        last_void = isfinite(s[i,:,k]).nonzero()[0][-1] + 1 # get first void above top filled site
                        # FIXME: THIS WILL DIE IF TOP HAS A VOID IN IT
                        v[i,j:last_void+1] += 1 #isnan(s[i,j:last_void+1,k])
                        s[i,j:last_void+1,k] = roll(s[i,j:last_void+1,k],1)
    return u,v,s

def close_voids(u,v,s):
    for i in range(nx):
        for j in arange(ny-1,-1,-1): # go from top to bottom
            for k in range(nm):
                if isnan(s[i,j,k]):
                    if random.rand() < 5e-2*dt/dy: # FIXME
                        v[i,j:] -= 1
                        s[i,j:,k] = roll(s[i,j:,k],-1)
    return u,v,s

def update_temperature(s,T,boundary):
    T[isnan(s)] = inlet_temperature # HACK
    T[boundary] = boundary_temperature
    T_inc = zeros_like(T)
    T_inc[1:-1,1:-1] = 1e-3*(T[2:,1:-1] + T[:-2,1:-1] + T[1:-1,2:] + T[1:-1,:-2] - 4*T[1:-1,1:-1])
    return T + T_inc

def get_average(s):
    s_bar = nanmean(nanmean(s,2),0)
    return s_bar
def get_hyperbolic_average(s):
    s_inv_bar = 1./nanmean(1./s,2)
    return s_inv_bar
def get_depth(s):
    depth = mean(mean(~isnan(s),axis=2),axis=1)
    return depth

nx = 20
ny = 20 #200
nm = 20
t_f = 2 # final time (s)
CFL = 0.2 # stability criteria, < 0.5
theta = 0.
H = 0.1 # m
s_M = 1e-3 # maximum particle size (m)
g = 9.81 # m/s^2
mode = 'bi'
# mode = 'poly'
# close = True
close = False
if mode == 'bi': s_m = 0.1
elif mode == 'poly': s_m = 0.2 # minimum size

temp_mode = sys.argv[1]
perf_plate = False
perf_pts = None
refill = False
internal_geometry = False

if temp_mode == 'hopper':
    nx = 51
    ny = 2*nx
    nm = 100
    cyclic_BC = False
    theta = 0.
    mu = float(sys.argv[2])
    half_width = int(sys.argv[3])
    IC_mode = 'top'
    top_fill = 0.7
    # IC_mode = 'random'
    # fill = 0.3
    save_inc = 100
    refill = True
    t_f = 200.
if temp_mode == 'temperature':
    nx = 21#201
    ny = 5*nx
    nm = 1
    cyclic_BC = False
    theta = 0.
    mu = float(sys.argv[2])
    half_width = int(sys.argv[3])
    IC_mode = 'top'
    top_fill = 0.8
    # IC_mode = 'random'
    # fill = 0.3
    save_inc = 50
    refill = True
    t_f = 20000.
    internal_geometry = True
    perf_rate = 0
    inlet_temperature = 400
    boundary_temperature = 20
elif temp_mode == 'WGCP':
    cyclic_BC = False
    perf_plate = True
    perf_rate = 0.01 # drop in probability across perf plate
    source_pts = [nx//6,nx//2,5*nx//6]
    perf_pts = [nx//3,2*nx//3]
    # u_applied = zeros_like(y)
    Tg_l = 0.01
    Tg_m = Tg_l/2.
    Tg_r = Tg_l/10.
    Tg = [Tg_l, Tg_m, Tg_r]
    gamma_dot = 0.
    half_width = 3
    IC_mode = 'top'
    top_fill = 0.7
    save_inc = 10
    mu = float(sys.argv[2])
elif temp_mode == 'slope':
    cyclic_BC = True
    mu = float(sys.argv[2])
    theta = float(sys.argv[3])
    Tg = float(sys.argv[4])
    ny = int(sys.argv[5])
    nm = 20
    nx = 5
    IC_mode = 'random'
    fill = 0.2
    save_inc = 10000
    t_f = 10.0 # s
elif temp_mode == 'collapse':
    cyclic_BC = False
    mu = float(sys.argv[2])
    theta = 0.
    # Tg = float(sys.argv[4])
    fill = 0.
    aspect_ratio = 1
    IC_mode = 'column'
    ny = 20#50
    nx = 4*ny
    nm = 50
    save_inc = 100
    t_f = 10.0 # s
elif temp_mode == 'mara':
    cyclic_BC = True
    mu = float(sys.argv[2])
    theta = 0.
    Tg = float(sys.argv[3])
    IC_mode = 'top'
    top_fill = 0.5
    ny = 50
    nx = ny*4
    nm = 5
    save_inc = 10
    t_f = 20.0 # s
elif temp_mode == 'pour':
    cyclic_BC = False
    mu = float(sys.argv[2])
    half_width = int(sys.argv[3])
    theta = 0.
    # Tg = float(sys.argv[4])
    fill = 0.
    ny = 100
    nx = 50
    nm = 1
    IC_mode = 'empty'
    save_inc = 1000
    t_f = 20.0 # s

if temp_mode == 'WGCP': folderName = 'plots/' + temp_mode + '/' + mode + '/' + str(Tg_l) + '/'
elif temp_mode == 'hopper': folderName = 'plots/' + temp_mode + '/mu_' + str(mu) + '/D_' + str(half_width) + '/'
elif temp_mode == 'slope': folderName = 'plots/' + temp_mode + '/' + mode + '/mu_' + str(mu) + '/theta_' + str(theta) + '/Tg_' + str(Tg) + '/ny_' + str(ny) + '/'
elif temp_mode == 'mara': folderName = 'plots/' + temp_mode + '/mu_' + str(mu) + '/Tg_' + str(Tg) + '/'
else: folderName = 'plots/' + temp_mode + '/mu_' + str(mu) + '/'
if not os.path.exists(folderName): os.makedirs(folderName)

fig = plt.figure(figsize=[nx/10.,ny/10.])
y = linspace(0,H,ny)
dy = y[1] - y[0]
x = linspace(-nx*dy/2,nx*dy/2,nx) # force equal grid spacing
X,Y = meshgrid(x,y,indexing='ij')

y += dy/2.
t = 0
s_avg_m = 1e-3 # average particle size in m
# t_p = s_avg_m/sqrt(g*H) # smallest confinement timescale (at bottom) (s)
free_fall_velocity = sqrt(g*s_avg_m)
P_scaling = (mu**2)/2.
# P_scaling = (mu**2)/sqrt(2)
if P_scaling > 1: # SOLVED: both swapping probabilities guaranteed to be less than or equal to 0.5
    P_adv = 0.5
    P_diff = P_adv/P_scaling
else:
    P_diff = 0.5
    P_adv = P_diff*P_scaling
# dt = P_adv*dy/free_fall_velocity

dt = 1.0

nt = int(ceil(t_f/dt))

s_bar_time = zeros([nt,ny])
nu_time = zeros_like(s_bar_time)
nu_time_x = zeros([nt,nx])
u_time = zeros_like(s_bar_time)

s = IC() # non-dimensional size
u = zeros([nx,ny])
v = zeros([nx,ny])

if temp_mode == 'WGCP':
    c = zeros_like(s) # original bin that particles started in
    c[perf_pts[0]:perf_pts[1]] = 1
    c[perf_pts[1]:] = 2
    c[isnan(s)] = nan

if internal_geometry:
    boundary = zeros([nx,ny],dtype=bool)
    # boundary[4:-4:5,:] = 1
    boundary[cos(500*2*pi*X) > 0] = 1
    boundary[:,:nx//2] = 0
    boundary[:,-nx//2:] = 0
    boundary[:,ny//2-5:ny//2+5] = 0
    boundary[abs(X)-2*half_width*dy > Y] = 1
    boundary[abs(X)-2*half_width*dy > H-Y] = 1
    boundary_tile = tile(boundary.T,[nm,1,1]).T
    s[boundary_tile] = nan
    # plt.pcolormesh(x,y,boundary.T)
    # plt.show()
    # sys.exit()

if temp_mode == 'temperature':
    T = inlet_temperature*ones_like(s)
    T[isnan(s)] = nan
    outlet_T = []

plot_s(x,y,s,folderName,t,perf_plate,perf_pts,s_m)
plot_nu(x,y,s,folderName,t,perf_plate,perf_pts,internal_geometry,boundary)
plot_u(x,y,s,u,v,folderName,t,nm,IC_mode,perf_plate,perf_pts,boundary)
if temp_mode == 'WGCP': plot_c(x,y,s,c,folderName,t,perf_plate,perf_pts)
if temp_mode == 'temperature': plot_T(x,y,s,T,folderName,t,boundary_temperature,inlet_temperature)
outlet = []

print('Running ' + folderName)
# print('Tg = ' + str(Tg) + ', k_add = ' + str(free_fall_velocity*dt/(Tg*H)) + '\n')
while t < nt:
    outlet.append(0)
    u = zeros_like(u)
    v = zeros_like(v)

    depth = get_depth(s)
    s_bar = get_average(s)
    s_inv_bar = get_hyperbolic_average(s)
    T = update_temperature(s,T,boundary)

    # u,v,s = move_holes(u,v,s)

    if t%2 == 0:
        u,v,s = move_holes_adv(u,v,s)
        u,v,s = move_holes_diff(u,v,s)
    else:
        u,v,s = move_holes_diff(u,v,s)
        u,v,s = move_holes_adv(u,v,s)

    u,v,s = add_temp(u,v,s)
    if close: u,v,s = close_voids(u,v,s)
    if t % save_inc == 0:
        plot_s(x,y,s,folderName,t,perf_plate,perf_pts,s_m)
        plot_nu(x,y,s,folderName,t,perf_plate,perf_pts,internal_geometry,boundary)
        plot_u(x,y,s,u,v,folderName,t,nm,IC_mode,perf_plate,perf_pts,boundary)
        # plot_outlet(outlet)
        if temp_mode == 'WGCP': plot_c(x,y,s,c,folderName,t,perf_plate,perf_pts)
        if temp_mode == 'hopper' or temp_mode =='temperature': savetxt(folderName + 'outlet.csv',array(outlet),delimiter=',')
        if temp_mode == 'temperature':
            plot_T(x,y,s,T,folderName,t,boundary_temperature,inlet_temperature)
            savetxt(folderName + 'outlet_T.csv',array(outlet_T),delimiter=',')
        if temp_mode == 'slope': savetxt(folderName + 'u.csv',u/sum(isnan(s),axis=2),delimiter=',')
        if temp_mode == 'collapse': plot_profile(x,nu_time_x,folderName,nt,t_f)

    s_bar_time[t] = s_bar # save average size
    u_time[t] = mean(u,axis=0)
    nu_time[t] = mean(1 - mean(isnan(s),axis=2),axis=0)
    nu_time_x[t] = mean(1 - mean(isnan(s),axis=2),axis=1)
    t += 1
    if t % 10 == 0: print(' t = ' + str(t*dt) + '                ', end='\r')
plot_s_bar(s_bar_time,nu_time,s_m,folderName)
plot_u_time(y,u_time,nu_time,folderName,nt)
save(folderName + 'nu_t_x.npy', nu_time_x)
if temp_mode == 'WGCP': plot_c(c)
if temp_mode == 'hopper': savetxt(folderName + 'outlet.csv',array(outlet),delimiter=',')
if temp_mode == 'slope': savetxt(folderName + 'u.csv',u/sum(isnan(s),axis=2),delimiter=',')
if temp_mode == 'collapse': plot_profile(x,nu_time_x,folderName,nt,t_f)
print('\nDone')
