#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 07:21:08 2022

@author: nwankwog
"""

#_________________________________________________________________
"Differentiation"
#_________________________________________________________________
import sys
import numpy as np
import matplotlib.pyplot as plt

def func(x):
    """generic function"""
    return np.cos(x)+x*np.sin(x)

def func_dot(x):
    """Derivative of the generic function"""
    return x*np.cos(x)



"Dispaly function and its derivative"
n_points = 1000     # number of points
#x_in = -6           # start 
#x_fin = -x_in       # symmetric domain
#x = np.linspace(x_in,x_fin,n_points) # independent variable
ini_x = -6
final_x = 6
x = np.linspace(ini_x,final_x,n_points)
y = func(x) #dependent variable
y_dot = func_dot(x) # derivative

fig1 = plt.figure()
plt.plot(x,y,'red')
plt.plot(x,y_dot,'blue')
plt.grid()
plt.xlabel('x',fontsize = 16)
plt.legend([r'$y$',r'$\dot y$'],fontsize=16)

#_____________________________________________________
#plot of true y, which is the correct solution
fig2 = plt.figure() #plot the correct solution 
plt.plot(x,y_dot,'magenta')
plt.grid()
plt.xlabel(r'$x$')
plt.ylabel(r'$\dot y$')
plt.legend([r'$\dot y$ truth'])

#_______________________________________________________

"Forward Finite Difference"
step_size = 0.25
x0 = ini_x                      # initialize first point
y_dot_forw = np.array([])      # initialize solution array 
x_forw = np.array([ini_x])      # initialize step points

while x0 <= final_x:
    new_x = func(x0)                              #f_k
    nextx = func(x0+step_size)                  #f_k+1
    slope = (nextx-new_x)/step_size     #(f_k+1 - f_k)/h
    x0 = x0 + step_size           
    x_forw = np.append(x_forw, x0)
    y_dot_forw = np.append(y_dot_forw, slope)
    
    
plt.plot(x_forw[:-1],y_dot_forw,'red')
plt.legend([r'$\dot y$ truth',r'$\dot y$ forward'])

#_________________________________________________________

"Backward Finite Difference"
x0 = ini_x                      # initialize first point
y_dot_back = np.array([])      # initialize solution array 
x_back = np.array([ini_x])      # initialize step points

while x0 <= final_x:
    new_x = func(x0)                              #f_k
    lastx = func(x0-step_size)                   #f_k-1
    slope = (new_x-lastx)/step_size      #(f_k - f_k-1)/h
    x0 = x0 + step_size
    x_back = np.append(x_back, x0)
    y_dot_back = np.append(y_dot_back, slope)
    
    
plt.plot(x_back[:-1],y_dot_back,'blue')
plt.legend([r'$\dot y$ truth',r'$\dot y$ forward',r'$\dot y$ backward'])

#________________________________________________________________________

"Central Finite Difference"
x0 = ini_x                      # initialize first point
y_dot_cent= np.array([])       # initialize solution array 
x_cent = np.array([ini_x])      # initialize step points

while x0 <= final_x:
    nextx = func(x0+step_size)                  #f_k+1
    lastx = func(x0-step_size)                   #f_k-1
    slope = (nextx-lastx)/(2*step_size)  #(f_k+1 - f_k-1)/2h
    x0 = x0 + step_size
    x_cent = np.append(x_cent, x0)
    y_dot_cent = np.append(y_dot_cent, slope)
    
    
plt.plot(x_cent[:-1],y_dot_cent,'green')
plt.legend([r'$\dot y$ truth',r'$\dot y$ forward',
            r'$\dot y$ backward',r'$\dot y$ central'])




#_________________________________________________________________
"Integration"
#_________________________________________________________________

"Auxiliarly functions"

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint 

def RHS(x, t):
    return -2*x


"Set the problem"
y0 = 3 # initial condition
t0 = 0 # initial time
tf = 2 # final time

"Evaluate exact solution"
time = np.linspace(t0,tf) # time spanned
y_true = odeint(RHS,y0,time) # solution

fig3 = plt.figure()
plt.plot(time,y_true,'k-',linewidth = 2)
plt.grid()
plt.xlabel('time')
plt.ylabel(r'$y(t)$')
plt.legend(['Truth'])

#___________________________________________________________

"Numerical integration"
step_size = 0.2 #value of the fixed step size

#____________________________________________________________
"First Order Runge-Kutta or Euler Method"
current_time = t0
timeline = np.array([t0])
current_value = y0
sol_rk1 = np.array([y0])

while current_time < tf-step_size:
    
    # Solve ODE
    slope = RHS(current_value, current_time)
    next_value = current_value + slope * step_size
    
    # Save Solution
    next_time = current_time + step_size
    timeline = np.append(timeline, next_time)
    sol_rk1 = np.append(sol_rk1, next_value)
    
    # Initialize Next Step
    current_time = next_time
    current_value = next_value
    
plt.plot(timeline,sol_rk1,'r-o',linewidth = 2)
plt.legend(['Truth','Runge-Kutta 1'])

#__________________________________________________________________
"Second Order Runge-Kutta"
current_time = t0
timeline = np.array([t0])
current_value = y0
sol_rk2 = np.array([y0])

while current_time < tf-step_size:
    
    # Solve ODE
    k1 = RHS(current_value, current_time)
    k2 = RHS(current_value + k1*step_size/2, current_time + step_size/2)
    next_value = current_value + k2 * step_size
    
    # Save Solution
    next_time = current_time + step_size
    timeline = np.append(timeline, next_time)
    sol_rk2 = np.append(sol_rk2, next_value)
    
    # Initialize Next Step
    current_time = next_time
    current_value = next_value
    
plt.plot(timeline,sol_rk2,'magenta',linewidth = 2)
plt.legend(['Truth','Runge-Kutta 1','Runge-Kutta 2'])

#___________________________________________________________________________
"Fourth Order Runge-Kutta"
current_time = t0
timeline = np.array([t0])
current_value = y0
sol_rk4 = np.array([y0])

while current_time < tf-step_size:
    
    # Solve ODE
    k1 = RHS(current_value, current_time)
    k2 = RHS(current_value + k1*step_size/2, current_time + step_size/2)
    k3 = RHS(current_value + k2*step_size/2, current_time + step_size/2)
    k4 = RHS(current_value + k3*step_size, current_time + step_size)
    next_value = current_value + (k1+2*k2+2*k3+k4) * step_size/6
    
    # Save Solution
    next_time = current_time + step_size
    timeline = np.append(timeline, next_time)
    sol_rk4 = np.append(sol_rk4, next_value)
    
    # Initialize Next Step
    current_time = next_time
    current_value = next_value
    
plt.plot(timeline,sol_rk4,'green',linewidth = 2)
plt.legend(['Truth','Runge-Kutta 1','Runge-Kutta 2','Runge-Kutta 4'])




#_________________________________________________________________
"Pendulum"
#_________________________________________________________________

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint 

def pendulum_free(x,t):
    """Dynamics of the pendulum without any constraint"""
    g = 9.81 # gravity constant
    l = 3 #length of pendulum
    xdot = np.zeros(2)
    xdot[0] = x[1]
    xdot[1] = -g/l*np.sin(x[0])
    return xdot

def pendulum_damped(x,t):
    """Dynamics of the pendulum with a damper"""
    g = 9.81 # gravity constant
    l = 3 #length of pendulum
    damp = 0.3 #damper coefficient
    xdot = np.zeros(2)
    xdot[0] = x[1]
    xdot[1] = -g/l*np.sin(x[0]) - damp*x[1]
    return xdot

def pendulum_controlled(x,t,u):
    """Dynamics of the pendulum without an actuator that gives control torque"""
    g = 9.81 # gravity constant
    l = 3 #length of pendulum
    m = 0.2 # mass of ball
    xdot = np.zeros(2)
    xdot[0] = x[1]
    xdot[1] = -g/l*np.sin(x[0]) + u/m/l/l
    return xdot

def RK1(func, y0, t):
    """Explicit Integrator Runge-Kutta Order 1"""
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i] #time step
        k1 = func(y[i], t[i]) #slope
        y[i+1] = y[i] + k1 * h #forward integration
    return y

def RK2(func, y0, t):
    """Explicit Integrator Runge-Kutta Order 2"""
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i] #times tep
        k1 = func(y[i], t[i])
        k2 = func(y[i] + k1 * h / 2., t[i] + h / 2)
        y[i+1] = y[i] + k2 * h
    return y

def RK4(func, y0, t):
    """Explicit Integrator Runge-Kutta Order 2"""
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i] #time - step
        k1 = func(y[i], t[i])
        k2 = func(y[i] + k1 * h / 2., t[i] + h / 2)
        k3 = func(y[i] + k2 * h / 2., t[i] + h / 2)
        k4 = func(y[i] + k3 * h, t[i] + h)
        y[i+1] = y[i] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
        return y
    
#________________________________________________________________
"Propagate Free Pendulum"
x0 = np.array([np.pi/3, 0])
t0 = 0.0
tf = 15.0
n_points = 1000
time = np.linspace(t0,tf,n_points)
y = odeint(pendulum_free,x0,time)
y_rk = RK4(pendulum_free,x0,time)

fig4 = plt.figure()
plt.subplot(2,1,1)
plt.plot(time,y[:,0],'b-',linewidth = 2)
plt.plot(time,y_rk[:,0],'c-.',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\theta$')
plt.legend(['odeint','rk4'])
plt.subplot(2,1,2)
plt.plot(time,y[:,1],'r-',linewidth = 2)
plt.plot(time,y_rk[:,1],'m-.',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\dot \theta$')
plt.legend(['odeint','rk4'])

#____________________________________________________________________
"Propagate Damped Pendulum"
x0 = np.array([np.pi/3, 0])
n_points = 25
time_new = np.linspace(t0,tf,n_points)
y = odeint(pendulum_damped,x0,time)
#y_rk1 = RK1(pendulum_damped,x0,time_new)
y_rk2 = RK2(pendulum_damped,x0,time_new)
y_rk4 = RK4(pendulum_damped,x0,time_new)

"Display"
fig5 = plt.figure()
plt.subplot(2,1,1)
plt.plot(time,y[:,0],'k-',linewidth = 2)
#plt.plot(time_new,y_rk1[:,0],'r-',linewidth = 2)
plt.plot(time_new,y_rk2[:,0],'b-',linewidth = 2)
plt.plot(time_new,y_rk4[:,0],'g-',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\theta$')
plt.subplot(2,1,2)
plt.plot(time,y[:,1],'k-',linewidth = 2)
#plt.plot(time_new,y_rk1[:,1],'r-',linewidth = 2)
plt.plot(time_new,y_rk2[:,1],'b-',linewidth = 2)
plt.plot(time_new,y_rk4[:,1],'g-',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\dot \theta$')

#________________________________________________________________________
"Add A PD controller "
Kp = 2  # PD Proportional gain
Kd = 2  # PD derivative gain

n_points = 100
delta_read = 0.01 # frequency of observations

"Control Pendulum To Stable Equilibrium [0,0]"
timeline = np.array([t0])  #initial value is initial condition
x_hyst = np.array([x0])  #initial value is initial condition

t_in = t0  #initialize time
x_in = x0  #initialize state

u = -Kp*x0[0] - Kd*x0[1]; #initial control

while t_in < tf:
    
    # Advance in time
    t_fin = t_in + delta_read
    
    # Solve ODE
    time = np.linspace(t_in,t_fin,n_points)
    y = odeint(pendulum_controlled,x_in,time,args=(u,))
    x_fin = y[-1,:]
    
    # Evaluate Control
    u = -Kp*x_fin[0] - Kd*x_fin[1];
    
    # Save Solution
    timeline = np.append(timeline, t_fin)
    x_hyst = np.vstack([x_hyst,x_fin]) #stack current state to solution vector
    
    # Initialize Next Timestep
    x_in = x_fin;
    t_in = t_fin;

# Display
fig6 = plt.figure()
plt.subplot(2,1,1)
plt.plot(timeline,x_hyst[:,0],'b-',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\theta$')
plt.subplot(2,1,2)
plt.plot(timeline,x_hyst[:,1],'b-',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\dot \theta$')

#_________________________________________________________________
"Control Pendulum To Untable Equilibrium [pi,0]"
Kp = 10  # PD Proportional gain
Kd = 5  # PD derivative gain

timeline = np.array([t0])
x_hyst = np.array([x0]);

t_in = t0  #initialize
x_in = x0

x_des = np.array([np.pi,0])

u = -Kp*(x0[0]-x_des[0]) - Kd*(x0[1]--x_des[1]) ; #initial control

while t_in < tf:
    
    # Advance in time
    t_fin = t_in + delta_read
    
    # Solve ODE
    time = np.linspace(t_in,t_fin,n_points)
    y = odeint(pendulum_controlled,x_in,time,args=(u,))
    x_fin = y[-1,:]
    
    # Evaluate Control
    u = -Kp*(x_fin[0]-x_des[0])  - Kd*(x_fin[1]-x_des[1]) ;
    
    # Save Solution
    timeline = np.append(timeline, t_fin)
    x_hyst = np.vstack([x_hyst,x_fin])
    
    # Initialize Next Timestep
    x_in = x_fin;
    t_in = t_fin;

# Display
fig7 = plt.figure()
plt.subplot(2,1,1)
plt.axhline(y=np.pi,linewidth = 0.5,color = 'r')
plt.plot(timeline,x_hyst[:,0],'b-',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\theta$')
plt.subplot(2,1,2)
plt.plot(timeline,x_hyst[:,1],'b-',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\dot \theta$')

#_____________________________________________________________________

"Control Pendulum To Untable Equilibrium [pi,0] and make it do a loop at t=7.5 seconds"
timeline = np.array([t0])
x_hyst = np.array([x0]);

t_in = t0  #initialize
x_in = x0

x_des_1 = np.array([np.pi,0])  #initial desired equilibrium state
x_des_2 = np.array([-np.pi,0]) # final desired equilibrium state
t_switch = 7.5 #time at which the pendulum rotates

u = -Kp*(x0[0]-x_des_1[0]) - Kd*(x0[1]--x_des_1[1]) ; #initial control

while t_in < tf:
    
    # Advance in time
    t_fin = t_in + delta_read
    
    # Solve ODE
    time = np.linspace(t_in,t_fin,n_points)
    y = odeint(pendulum_controlled,x_in,time,args=(u,))
    x_fin = y[-1,:]
    
    # Evaluate Control
    if t_fin <= t_switch:
        u = -Kp*(x_fin[0]-x_des_1[0])  - Kd*(x_fin[1]-x_des_1[1]) 
    else:
        u = -Kp*(x_fin[0]-x_des_2[0])  - Kd*(x_fin[1]-x_des_2[1]) 
    
    # Save Solution
    timeline = np.append(timeline, t_fin)
    x_hyst = np.vstack([x_hyst,x_fin])
    
    # Initialize Next Timestep
    x_in = x_fin;
    t_in = t_fin;

# Display
fig8 = plt.figure()
plt.subplot(2,1,1)
plt.axhline(y=np.pi,linewidth = 0.5,color = 'r')
plt.axhline(y=-np.pi,linewidth = 0.5,color = 'r')
plt.plot(timeline,x_hyst[:,0],'b-',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\theta$')
plt.subplot(2,1,2)
plt.plot(timeline,x_hyst[:,1],'b-',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\dot \theta$')





#_________________________________________________________________
"Lorenz63 System"
#_________________________________________________________________

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint 

def Lorenz63(x,t,sigma,rho,beta):
    """The Lorenz63 system for different coefficients"""
    xdot = np.zeros(3)
    xdot[0] = sigma*(x[1]-x[0]);
    xdot[1] = x[0]*(rho-x[2])-x[1];
    xdot[2] = x[0]*x[1] - beta*x[2];
    return xdot


"Set parameters"
rho = 28
sigma = 10
beta = 8/3

x0 = 5*np.ones(3); #initial state
t_in = 0 #initial time
t_fin = 20 #final time
n_points = 1000
time = np.linspace(t_in,t_fin,n_points) #time vector

# Solve ODE
y = odeint(Lorenz63,x0,time,args = (sigma,rho,beta)) #propagation

# Display
fig9 = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(y[:,0], y[:,1], y[:,2],'b') #3d plottingsolution

#_________________________________________________________________
"Random Initial Condition"
n_ic = 20 #number of initial conditions
x0s = np.zeros((3,n_ic))

#Initial condition domain
x0s[0,:] = 20 * 2 * (np.random.rand(n_ic) - 0.5);
x0s[1,:] = 30 * 2 * (np.random.rand(n_ic) - 0.5);
x0s[2,:] = 50 * np.random.rand(n_ic);


fig10 = plt.figure()
ax = plt.axes(projection='3d')
for i in range(n_ic):
    y = odeint(Lorenz63,x0s[:,i],time,args = (sigma,rho,beta)) #propagation
    ax.plot3D(y[:,0], y[:,1], y[:,2]) #visualization
    
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')



"The end of DAY 6 class"