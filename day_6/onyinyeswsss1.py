#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 11:47:14 2022

@author: nwankwog
"""
#SWSSS Day 1 question 1 solution

import numpy as np
import sys
import matplotlib.pylab as plt

def func(x):
    return np.cos(x) + x*np.sin(x)

def func_dot(x):
    return x*np.cos(x)

n_points = 1000 #number of points i want to be in between -6 and 6
x = np.linspace(-6, 6, n_points)
y = func(x)
y_dot = func_dot(x)

fig = plt.figure(figsize=(15,15))
plt.plot(x, y, 'green')
plt.plot(x, y_dot, 'magenta')


#________________________________________________________________
import numpy as np
import sys
import matplotlib.pylab as plt
from scipy.integrate import odeint

def RHS(x, t):
    return -2*x

#print("start")
y_x0 = 3
ini_t = 0
tf = 2
time = np.linspace(ini_t, tf)
soln_y = odeint(RHS, y_x0, time)

fig = plt.figure()
plt.plot(time,soln_y, 'blue', label='True Y') 
plt.legend()

#_________________________
#now, lets do initialization
step_size= 0.2
"First order Runge-Kutta 1st order or Eular Method"

timeline = np.array([ini_t])
soln_y1 = np.array([y_x0])
x_appr = timeline
y_appr = soln_y1
y = y_x0
#print(soln_y1)

while ini_t <= tf-step_size: # (tf-step_size) to not finsih one step after


    #evaluation
    f_RHS1 = RHS(y,ini_t)*step_size  #h*f'_k
    y1 = y + f_RHS1 #f_k+1

    #storage

    ini_t= ini_t+step_size
    x_appr = np.append(x_appr,ini_t) #approximation for the integration in x
    y_appr = np.append(y_appr,y1)  #approximation for the integration in y

    #iteration (intialize next step)

    y=y1
    t=ini_t

   

plt.plot(x_appr,y_appr,'red',linewidth=2) # plotting x and x from the approximation function
plt.legend(['Y truth', 'Y appro'],fontsize=16)


#_________________________
#now, lets do initialization
step_size= 0.2
"Second order Runge-Kutta 2nd order or Eular Method"

timeline = np.array([ini_t])
soln_y1 = np.array([y_x0])
x_appr = timeline
y_appr = soln_y1
y = y_x0
#print(soln_y1)

while ini_t <= tf-step_size: # (tf-step_size) to not finsih one step after


    #evaluation
    f_RHS1 = y + RHS(y,ini_t)*(step_size)/2  #h*f'_k
    f_RHS2 = ini_t + (step_size/2)
    f_RHS3 = step_size*(RHS(f_RHS1,f_RHS2))
    y1 = y + f_RHS3

    ini_t= ini_t+step_size
    x_appr = np.append(x_appr,ini_t) #approximation for the integration in x
    y_appr = np.append(y_appr,y1)  #approximation for the integration in y

    #iteration (intialize next step)

    y=y1
    t=ini_t

   

plt.plot(x_appr,y_appr,'magenta',linewidth=2) # plotting x and x from the approximation function
plt.legend(['Y truth', 'Y 1st appro', 'Y 2nd appro'],fontsize=16)
#_____________________________________________________________
#now, lets do initialization
step_size= 0.2
"Fourth order Runge-Kutta 4th order or Eular Method"

timeline = np.array([ini_t])
soln_y2 = np.array([y_x0])
x_appr = timeline
y_appr = soln_y2
y = y_x0
#print(soln_y1)

while ini_t <= tf-step_size: # (tf-step_size) to not finsih one step after


    #evaluation
    f_RHS1 = RHS(y,ini_t)
    f_RHS2 = RHS(y+ f_RHS1*step_size/2, ini_t + step_size/2)
    f_RHS3 = RHS(y+ f_RHS2*step_size/2, ini_t + step_size/2)
    f_RHS4 = RHS(y+ f_RHS3*step_size/2, ini_t + step_size/2)
    y1 = y + (f_RHS1+2*f_RHS2+2*f_RHS3+f_RHS4)

    new_time= ini_t+step_size
    x_appr = np.append(x_appr,new_time) #approximation for the integration in x
    y_appr = np.append(y_appr,y1)  #approximation for the integration in y

    #iteration (intialize next step)

    y=y1
    t=ini_t

   

plt.plot(x_appr,y_appr,'black',linewidth=2) # plotting x and x from the approximation function
#plt.legend(['Y truth', 'Y 1st appro', 'Y 2nd appro'],fontsize=16)

#________________________________________________________________

import numpy as np
import sys
import matplotlib.pylab as plt
from scipy.integrate import solve_ivp, odeint


def pendulum_f(theta, t):
    l = 3 #meters
    g = 9.81 #m/s
    thetadot = np.zeros(2)
    thetadot[0] = theta[1]
    thetadot[1] = -g / l * np.sin(theta[0])
    return thetadot
    
def pendulum_d(theta, t):
    l = 3 #meters
    g = 9.81 #m/s
    d = 0.3
    thetadot = np.zeros(2)
    thetadot[0] = theta[1]
    thetadot[1] = -g / l * np.sin(theta[0]) - d*theta[1]
    return thetadot

#for free pendulum
time = np.linspace(0, 15, 1000)
x0 = np.array([np.pi/3, 0])
y = odeint(pendulum_f, x0, time)
y1 = odeint(pendulum_d, x0, time)

fig, axs= plt.subplots(2)
axs[0].plot(time, y)
axs[1].plot(time, y1)
#________________________________________________________________
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

sigma=10
ro=28
beta=8/3
t0=0
tf=20
t = np.linspace(t0,tf,1000)
x = 5 * np.ones(3)  
 

def lorenz63(x,t,sigma,ro,beta):
    xdot = sigma*(x[1]-x[0])
    ydot = x[0]*(ro-x[2])-x[1]
    zdot = (x[0]*x[1]) - (beta*x[2])
    return xdot, ydot, zdot

 

solu_lorenz=odeint(lorenz63,x,t,args=(sigma,ro,beta))
fid1 = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(solu_lorenz[:,0],solu_lorenz[:,1],solu_lorenz[:,2],'b')

#________________________________________________________________