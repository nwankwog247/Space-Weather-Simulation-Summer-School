#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 17:54:07 2022

@author: nwankwog
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi
%matplotlib qt
plt.close()
import matplotlib.animation as animation

"Flow parameters"
nu = 0.01
c = -2
u0 = 0

"Scheme parameters"
beta = 0

"Number of points"
N = 32
Dx = 1/N
x = np.linspace(0,1,N+1)
xx = np.concatenate(([x[0]-Dx],x))


"Time parameters"
delta_t = 1/50
time = np.arange(0,3+delta_t,delta_t)
numbt = np.size(time)

"Initialize solution variable"
U = np.zeros((N+1,numbt))


for it in range(numbt-1):

    "System matrix and RHS term"
    "Diffusion term"
    Diff = nu*(1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))

    "Advection term:"
        
    "Sensor"
    U0 = U[:,it]
    uaux = np.concatenate(([U0[0]], U0,[U0[N]]))
    Du = uaux[1:N+3] - uaux[0:N+2] + 1e-8
    r = Du[0:N+1]/Du[1:N+2]
    
    "Limiter"
    if beta>0:
        phi = np.minimum(np.minimum(beta*r,1),np.minimum(r,beta))
        phi = np.maximum(0,phi)
    else:
        phi = 2*r/(r**2 + 1)
        
    phim = phi[0:N]
    phip = phi[1:N+1]
        
    
    "Upwind scheme"
    cp = np.max([c,0])
    cm = np.min([c,0])
    
    Advp = cp*(np.diag(1-phi) - np.diag(1-phip,-1))
    Advm = cm*(np.diag(1-phi) - np.diag(1-phim,1))
    Alow = Advp-Advm
    
    "Centered differences"
    Advp = -0.5*c*np.diag(phip,-1)
    Advm = -0.5*c*np.diag(phim,1)
    Ahigh = Advp-Advm
        
    Adv = (1/Dx)*(Ahigh + Alow)
    A = Diff + Adv
    
    "Source term"
    sine = np.sin(2*pi*time[it+1])
    sineplus = 0.5*(sine + np.abs(sine))
    F = 100*np.exp(-((x-0.8)/0.01)**2)*sineplus
    
    "Temporal terms"
    A = A + (1/delta_t)*np.diag(np.ones(N+1))
    F = F + U0/delta_t

    "Boundary condition at x=0"
    A[0,:] = (1/Dx)*np.concatenate(([1.5, -2, 0.5],np.zeros(N-2)))
    F[0] = 0

    "Boundary condition at x=1"
    A[N,:] = np.concatenate((np.zeros(N),[1]))
    F[N] = u0


    "Solution of the linear system AU=F"
    u = np.linalg.solve(A,F)
    U[:,it+1] = u
    u = u[0:N+1]


"Animation of the results"
fig = plt.figure()
ax = plt.axes(xlim =(0, 1),ylim =(u0-1e-2,u0+0.5)) 
myAnimation, = ax.plot([], [],':ob',linewidth=2)
plt.grid()
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

def animate(i):
    
    u = U[0:N+1,i]
    plt.plot(x,u)
    myAnimation.set_data(x, u)
    return myAnimation,

anim = animation.FuncAnimation(fig,animate,frames=range(1,numbt),blit=True,repeat=False)


if nu>0:
    "Peclet number"
    P = np.abs(c*Dx/nu)
    print("Pe number Pe=%g\n" % P);

"CFL number"
CFL = np.abs(c*delta_t/Dx)
print("CFL number CFL=%g\n" % CFL);

#___________________________________________________________________
"Flow parameters"
nu = 0.0001
c = -2
u0 = 0

"Scheme parameters"
beta = 0

"Number of points"
N = 32
Dx = 1/N
x = np.linspace(0,1,N+1)
xx = np.concatenate(([x[0]-Dx],x))


"Time parameters"
delta_t = 1/50
time = np.arange(0,3+delta_t,delta_t)
numbt = np.size(time)

"Initialize solution variable"
U = np.zeros((N+1,numbt))


for it in range(numbt-1):

    "System matrix and RHS term"
    "Diffusion term"
    Diff = nu*(1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))

    "Advection term:"
        
    "Former U, we defined"
    U0 = U[:,it]

    "Upwind scheme"
    cp = np.max([c,0])
    cm = np.min([c,0])
    
    Advp = cp*(np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1))
    Advm = cm*(np.diag(np.ones(N+1)) - np.diag(np.ones(N),1))  
    Adv = (1/Dx)*(Advp - Advm)
    A = Diff + Adv
    
    "Source term"
    sine = np.sin(2*pi*time[it+1])
    sineplus = 0.5*(sine + np.abs(sine))
    F = 100*np.exp(-((x-0.8)/0.01)**2)*sineplus
    
    "Temporal terms"
    A = A + (1/delta_t)*np.diag(np.ones(N+1))
    F = F + U0/delta_t

    "Boundary condition at x=0"
    A[0,:] = (1/Dx)*np.concatenate(([1.5, -2, 0.5],np.zeros(N-2)))
    F[0] = 0

    "Boundary condition at x=1"
    A[N,:] = np.concatenate((np.zeros(N),[1]))
    F[N] = u0


    "Solution of the linear system AU=F"
    u = np.linalg.solve(A,F)
    U[:,it+1] = u
    u = u[0:N+1]


"Animation of the results"
fig2 = plt.figure()
ax = plt.axes(xlim =(0, 1),ylim =(u0-1e-2,u0+0.5)) 
myAnimation, = ax.plot([], [],':ob',linewidth=2)
plt.grid()
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

def animate(i):
    
    u = U[0:N+1,i]
    plt.plot(x,u)
    myAnimation.set_data(x, u)
    return myAnimation,

anim = animation.FuncAnimation(fig2,animate,frames=range(1,numbt),blit=True,repeat=False)


if nu>0:
    "Peclet number"
    P = np.abs(c*Dx/nu)
    print("Pe number Pe=%g\n" % P);

"CFL number"
CFL = np.abs(c*delta_t/Dx)
print("CFL number CFL=%g\n" % CFL);


#___________________________________________________________________
"Flow parameters"
nu = 0.0001
c = -2
u0 = 0

"Scheme parameters"
#beta = 0

"Number of points"
N = 32
Dx = 1/N
x = np.linspace(0,1,N+1)
xx = np.concatenate(([x,x[-1]+Dx]))
order = 2

"Time parameters"
delta_t = 1/50
time = np.arange(0,3+delta_t,delta_t)
numbt = np.size(time)

"Initialize solution variable"
U = np.zeros((N+1,numbt))


for it in range(numbt-1):

    "System matrix and RHS term"
    "Diffusion term"
    Diff = nu*(1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))

    "Advection term:"
        
    "Former U, we defined"
    U0 = U[:,it]

    "Upwind scheme"
    cp = np.max([c,0])
    cm = np.min([c,0])
    
    if order<2:
        Advp = cp*(np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1))
        Advm = cm*(np.diag(np.ones(N+1)) - np.diag(np.ones(N),1))
    else:
        Advp = cp*(np.diag(1.5**np.ones(N+1)) - np.diag(np.ones(N),-1))
        Advm = cm*(np.diag(1.5**np.ones(N+1)) - np.diag(np.ones(N),1))
    
    Adv = (1/Dx)*(Advp - Advm)
    A = Diff + Adv
    
    "Source term"
    sine = np.sin(2*pi*time[it+1])
    sineplus = 0.5*(sine + np.abs(sine))
    F = 100*np.exp(-((x-0.8)/0.01)**2)*sineplus
    
    "Temporal terms"
    A = A + (1/delta_t)*np.diag(np.ones(N+1))
    F = F + U0/delta_t

    "Boundary condition at x=0"
    A[0,:] = (1/Dx)*np.concatenate(([1.5, -2, 0.5],np.zeros(N-2)))
    F[0] = 0

    "Boundary condition at x=1"
    A[N,:] = np.concatenate((np.zeros(N),[1]))
    F[N] = u0


    "Solution of the linear system AU=F"
    u = np.linalg.solve(A,F)
    U[:,it+1] = u
    u = u[0:N+1]


"Animation of the results"
fig3 = plt.figure()
ax = plt.axes(xlim =(0, 1),ylim =(u0-1e-2,u0+0.5)) 
myAnimation, = ax.plot([], [],':ob',linewidth=2)
plt.grid()
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

def animate(i):
    
    u = U[0:N+1,i]
    plt.plot(x,u)
    myAnimation.set_data(x, u)
    return myAnimation,

anim = animation.FuncAnimation(fig3,animate,frames=range(1,numbt),blit=True,repeat=False)


if nu>0:
    "Peclet number"
    P = np.abs(c*Dx/nu)
    print("Pe number Pe=%g\n" % P);

"CFL number"
CFL = np.abs(c*delta_t/Dx)
print("CFL number CFL=%g\n" % CFL);

#___________________________________________________________________
"Flow parameters"
nu = 0.0001
c = -2
u0 = 0

"Scheme parameters"
#beta = 0

"Number of points"
N = 32
Dx = 1/N
x = np.linspace(0,1,N+1)
xx = np.concatenate(([x[0]-Dx],x))
order = 2

"Time parameters"
delta_t = 1/50
time = np.arange(0,3+delta_t,delta_t)
numbt = np.size(time)

"Initialize solution variable"
U = np.zeros((N+1,numbt))


for it in range(numbt-1):

    "System matrix and RHS term"
    "Diffusion term"
    Diff = nu*(1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))

    "Advection term:"
        
    "Former U, we defined"
    U0 = U[:,it]

    "Upwind scheme"
    cp = np.max([c,0])
    cm = np.min([c,0])
    
    if order<2:
        Advp = cp*(np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1))
        Advm = cm*(np.diag(np.ones(N+1)) - np.diag(np.ones(N),1))
    else:
        Advp = cp*(np.diag(1.5**np.ones(N+2)) + np.diag(-2*np.ones(N+1),-1) + np.diag(0.5*np.ones(N),2))
        Advm = cm*(np.diag(1.5**np.ones(N+2)) + np.diag(-2*np.ones(N+1),1) + np.diag(0.5*np.ones(N),2))
    
    
    Adv = (1/Dx)*(Advp - Advm)
    A = Diff + Adv
    
    "Source term"
    sine = np.sin(2*pi*time[it+1])
    sineplus = 0.5*(sine + np.abs(sine))
    F = 100*np.exp(-((x-0.8)/0.01)**2)*sineplus
    
    "Temporal terms"
    A = A + (1/delta_t)*np.diag(np.ones(N+1))
    F = F + U0/delta_t

    "Boundary condition at x=0"
    A[0,:] = (1/Dx)*np.concatenate(([1.5, -2, 0.5],np.zeros(N-2)))
    F[0] = 0

    "Boundary condition at x=1"
    A[N,:] = np.concatenate((np.zeros(N),[1]))
    F[N] = u0


    "Solution of the linear system AU=F"
    u = np.linalg.solve(A,F)
    U[:,it+1] = u
    u = u[0:N+1]


"Animation of the results"
fig4 = plt.figure()
ax = plt.axes(xlim =(0, 1),ylim =(u0-1e-2,u0+0.5)) 
myAnimation, = ax.plot([], [],':ob',linewidth=2)
plt.grid()
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

def animate(i):
    
    u = U[0:N+1,i]
    plt.plot(x,u)
    myAnimation.set_data(x, u)
    return myAnimation,

anim = animation.FuncAnimation(fig4,animate,frames=range(1,numbt),blit=True,repeat=False)


if nu>0:
    "Peclet number"
    P = np.abs(c*Dx/nu)
    print("Pe number Pe=%g\n" % P);

"CFL number"
CFL = np.abs(c*delta_t/Dx)
print("CFL number CFL=%g\n" % CFL);