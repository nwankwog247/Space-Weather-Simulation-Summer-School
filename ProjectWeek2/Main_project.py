#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 10:09:07 2022

@author: aliaaafify
"""
# numpy for fast arrays and some linear algebra.
import numpy as np 
# import matplotlib for plotting
import matplotlib.pyplot as plt
# import Dormand-Prince Butcher tableau
import Matrix_AO as dp
# import our Runge-Kutta integrators
import projrectB_week2 as rk
#import intial condion of temp_in_k
from radtran import *
#import animation
import matplotlib.animation as animation

def dormand_prince_integrator(f, x, t, h):
    return rk.explicit_RK_stepper(f, x, t, h, dp.a, dp.b, dp.c)
    #return ... # please complete this function
                # so it returns the prediction for the 
                # Dormand-Prince method 
                # To that end, use rk.explicit_rk_stepper!

# Feel free play around with the following quantities 
# and see how the solution changes!
#-----------------------------------------------------------------------------
# Initialize temperature as a function of altitude
#-----------------------------------------------------------------------------

def init_temp(alt_in_km):
    temp_in_k = 200 + 600 * np.tanh( (alt_in_km - 100) / 100.0)
    return temp_in_k

# time horizon
tspan = (0.0,1439.0)
time_points_analytical = np.linspace(tspan[0],tspan[1], 1439)
# time step 
h = 5.0
# initial condition
#x_0 = 3.0
# initialize altitudes (step 2):

nAlts = 41
alts = np.linspace(100, 500, num = nAlts)
print('alts : ', alts)

# initial condition
temp = init_temp(alts)
print('temp : ', temp)

########################################
### hereafter no more code modification necessary
########################################

# model right-hand-side
def f(temp,time_points_analytical):
    
    #for i in range(time_points_analytical):
        
    Qeuv ,rho, dTdt = ionosphere(temp)
    #print(Qeuv/(rho*1500))
        
    return -10*Qeuv/(rho*1500)

# simulate model 
trajectory, time_points = rk.integrate(f, # ODE right-hand-side
                                         temp, # initial condition
                                         tspan, # time horizon
                                         h, # time step 
                                         dormand_prince_integrator) # integrator

# analytical solution
time_points_analytical = np.linspace(tspan[0],tspan[1], 1439)

#trajectory_analytical = temp * np.exp(f(temp,time_points_analytical)*time_points_analytical)

# plot trace
# fig, ax = plt.subplots()
# ax.set_xlabel("time")
# ax.set_ylabel("x(t)")
# ax.plot(time_points, trajectory, linewidth=2, color="red", marker="o")
# #ax.plot(time_points_analytical, trajectory_analytical, linewidth=2, color="black", linestyle="dashed")
# fig.savefig("temp_traj.pdf")


# fig, ax = plt.subplots()
# ax.set_xlabel("T(z) [K]")
# ax.set_ylabel("z [km]")
# for i in range(0, len(trajectory), 20):
#     ax.plot(trajectory[i], alts , linewidth=2, color="red")

# plt.show()

"Animation of the results"
fig = plt.figure()
ax = plt.axes(xlim =(100, 1000),ylim =(80,600))
myAnimation, = ax.plot([], [],':ob',linewidth=2)
plt.grid()
plt.xlabel("T(z) [K]",fontsize=16)
plt.ylabel("z [km]",fontsize=16)

def animate(i):
    
    #u = U[0:N+1,i]
    plt.plot(trajectory[i], alts, linewidth=2, color="red")
    myAnimation.set_data(trajectory[i], alts)
    return myAnimation,

anim = animation.FuncAnimation(fig,animate,frames=range(0, len(trajectory), 5),blit=True,repeat=False)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

anim.save('projectB.mp4', writer=writer)
