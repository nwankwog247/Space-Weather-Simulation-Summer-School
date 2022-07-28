#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 15:18:49 2022

@author: nwankwog
"""

#_________________________________________________________________
"Begining of Afternoon code"
#_________________________________________________________________
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation    #We have to load this
from math import pi
#%matplotlib qt
plt.close()


#____________________________________________________________________
"Flow parameters"
nu = 0.01
c = 2

"Number of points"
N = 128
Dx = 1/N
x = np.linspace(0,1,N+1)

"System matrix and RHS term"
"Diffusion term"
Diff = nu*(1/Dx**2)*(2*np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1) - np.diag(np.ones(N-2),1))
"Advection term: centered differences"
Advp = -0.5*c*np.diag(np.ones(N-2),-1)
Advm = -0.5*c*np.diag(np.ones(N-2),1)
Adv = (1/Dx)*(Advp-Advm)
A = Diff + Adv
"Source term"
F = np.ones(N-1)


"Solution of the linear system AU=F"
U = np.linalg.solve(A,F)
u = np.concatenate(([0],U,[0]))
ua = (1/c)*(x-((1-np.exp(c*x/nu))/(1-np.exp(c/nu))))


"Plotting solution"
fig1 = plt.figure()
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
plt.legend(fontsize=12,loc='upper left')
plt.grid()
plt.axis([0, 1,0, 2/c])
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)


"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)

"Peclet number"
P = np.abs(c*Dx/nu)
print("Pe number Pe=%g\n" % P);

#_______________________________________________________________

"Number of points"
N = 256
Dx = 1/N
x = np.linspace(0,1,N+1)

"System matrix and RHS term"
"Diffusion term"
Diff = nu*(1/Dx**2)*(2*np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1) - np.diag(np.ones(N-2),1))
"Advection term: centered differences"
Advp = -0.5*c*np.diag(np.ones(N-2),-1)
Advm = -0.5*c*np.diag(np.ones(N-2),1)
Adv = (1/Dx)*(Advp-Advm)
A = Diff + Adv
"Source term"
F = np.ones(N-1)


"Solution of the linear system AU=F"
U = np.linalg.solve(A,F)
u = np.concatenate(([0],U,[0]))
ua = (1/c)*(x-((1-np.exp(c*x/nu))/(1-np.exp(c/nu))))


"Plotting solution"
fig2 = plt.figure()
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
plt.legend(fontsize=12,loc='upper left')
plt.grid()
plt.axis([0, 1,0, 2/c])
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)


"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)

"Peclet number"
P = np.abs(c*Dx/nu)
print("Pe number Pe=%g\n" % P);

#_______________________________________________________________
"Number of points"
N = 512
Dx = 1/N
x = np.linspace(0,1,N+1)

"System matrix and RHS term"
"Diffusion term"
Diff = nu*(1/Dx**2)*(2*np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1) - np.diag(np.ones(N-2),1))
"Advection term: centered differences"
Advp = -0.5*c*np.diag(np.ones(N-2),-1)
Advm = -0.5*c*np.diag(np.ones(N-2),1)
Adv = (1/Dx)*(Advp-Advm)
A = Diff + Adv
"Source term"
F = np.ones(N-1)


"Solution of the linear system AU=F"
U = np.linalg.solve(A,F)
u = np.concatenate(([0],U,[0]))
ua = (1/c)*(x-((1-np.exp(c*x/nu))/(1-np.exp(c/nu))))


"Plotting solution"
fig3 = plt.figure()
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
plt.legend(fontsize=12,loc='upper left')
plt.grid()
plt.axis([0, 1,0, 2/c])
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)


"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)

"Peclet number"
P = np.abs(c*Dx/nu)
print("Pe number Pe=%g\n" % P);
#_______________________________________________________________

"Number of points"

N = 32
Dx = 1/N
x = np.linspace(0,1,N+1)

"System matrix and RHS term"
"Diffusion term"
Diff = nu*(1/Dx**2)*(2*np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1) - np.diag(np.ones(N-2),1))
"Advection term: centered differences"
Advp = -0.5*c*np.diag(np.ones(N-2),-1)
Advm = -0.5*c*np.diag(np.ones(N-2),1)
Adv = (1/Dx)*(Advp-Advm)
A = Diff + Adv
"Source term"
F = np.ones(N-1)


"Solution of the linear system AU=F"
U = np.linalg.solve(A,F)
u = np.concatenate(([0],U,[0]))
ua = (1/c)*(x-((1-np.exp(c*x/nu))/(1-np.exp(c/nu))))


"Plotting solution"
fig4 = plt.figure()
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
plt.legend(fontsize=12,loc='upper left')
plt.grid()
plt.axis([0, 1,0, 2/c])
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)


"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)

"Peclet number"
P = np.abs(c*Dx/nu)
print("Pe number Pe=%g\n" % P);

#_______________________________________________________________

"Number of points"

N = 64
Dx = 1/N
x = np.linspace(0,1,N+1)
cp = max(c,0) #c plus
cm = min(c,0) #c minus
order = 2

"System matrix and RHS term"
"Diffusion term"
Diff = nu*(1/Dx**2)*(2*np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1) - np.diag(np.ones(N-2),1))

#here, i will take note of the if condition for the advection term
if order<2:
    "Advection term: first order upwind"
    Advp = cp*(np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1)) # A advective plus - Advp
    Advm = cm*(np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),1))  # A advective minus - Advm
else:
    "Advection term: second order centered differences"
    Advp = -0.5*c*np.diag(np.ones(N-2),-1)
    Advm = -0.5*c*np.diag(np.ones(N-2),1)
    
    
Adv = (1/Dx)*(Advp-Advm)
A = Diff + Adv

"Source term"
F = np.ones(N-1)

"Solution of the linear system AU=F"
U = np.linalg.solve(A,F)
u = np.concatenate(([0],U,[0]))
ua = (1/c)*(x-((1-np.exp(c*x/nu))/(1-np.exp(c/nu))))


"Plotting solution"
fig5 = plt.figure()
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
plt.legend(fontsize=12,loc='upper left')
plt.grid()
plt.axis([0, 1,0, 2/c])
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)


"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)

"Peclet number"
P = np.abs(c*Dx/nu)
print("Pe number Pe=%g\n" % P);

#_________________________________________________________________________
"Number of points"

N = 64
Dx = 1/N
x = np.linspace(0,1,N+1)
cp = max(c,0) #c plus
cm = min(c,0) #c minus
order = 1

"System matrix and RHS term"
"Diffusion term"
Diff = nu*(1/Dx**2)*(2*np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1) - np.diag(np.ones(N-2),1))

#here, i will take note of the if condition for the advection term
if order<2:
    "Advection term: first order upwind"
    Advp = cp*(np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1)) # A advective plus - Advp
    Advm = cm*(np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),1))  # A advective minus - Advm
else:
    "Advection term: second order centered differences"
    Advp = -0.5*c*np.diag(np.ones(N-2),-1)
    Advm = -0.5*c*np.diag(np.ones(N-2),1)
    
    
Adv = (1/Dx)*(Advp-Advm)
A = Diff + Adv

"Source term"
F = np.ones(N-1)

"Solution of the linear system AU=F"
U = np.linalg.solve(A,F)
u = np.concatenate(([0],U,[0]))
ua = (1/c)*(x-((1-np.exp(c*x/nu))/(1-np.exp(c/nu))))


"Plotting solution"
fig6 = plt.figure()
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
plt.legend(fontsize=12,loc='upper left')
plt.grid()
plt.axis([0, 1,0, 2/c])
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)


"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)

"Peclet number"
P = np.abs(c*Dx/nu)
print("Pe number Pe=%g\n" % P);

#_________________________________________________________________________
import numpy as np
import matplotlib.pyplot as plt
from math import pi
%matplotlib qt
plt.close()
import matplotlib.animation as animation

"Flow parameters"
nu = 0.01
c = 2

"Scheme parameters"
beta = 1

"Number of points"

N = 64
Dx = 1/N
x = np.linspace(0,1,N+1)

order = 1

"Time parameters"
delta_t = 0.1
time = np.arange(0,3+delta_t,delta_t)
numbt = len(time)

"Initialize U here"
U = np.zeros((N-1,numbt))

for it in range(numbt-1):
    "Diffusion term"
    Diff = nu*(1/Dx**2)*(2*np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1) - np.diag(np.ones(N-2),1))

    #here, i will take note of the if condition for the advection term
    if order<2:
        cp = np.max([c,0]) #c plus
        cm = np.min([c,0]) #c minus
        "Advection term: first order upwind"
        Advp = cp*(np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1)) # A advective plus - Advp
        Advm = cm*(np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),1))  # A advective minus - Advm
    else:
        "Advection term: second order centered differences"
        Advp = -0.5*c*np.diag(np.ones(N-2),-1)
        Advm = -0.5*c*np.diag(np.ones(N-2),1)
    
    Adv = (1/Dx)*(Advp-Advm)
    A = Diff + Adv
    
    "Source term"
    F = np.ones(N-1)
    
    "Temporal terms"
    U0 = U[:,it]
    A = A + (1/delta_t)*np.diag(np.ones(N-1))
    F = F + U0/delta_t
    
    "Solution of the linear system AU=F"
    u = np.linalg.solve(A,F)
    U[:,it+1] = u
    
u = np.concatenate(([0],u,[0]))

ua = (1/c)*(x-((1-np.exp(c*x/nu))/(1-np.exp(c/nu))))

"Animation of the results"
fig7 = plt.figure()
ax = plt.axes(xlim =(0, 1),ylim =(0,1/c)) 
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
myAnimation, = ax.plot([], [],':ob',linewidth=2)
plt.grid()
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

def animate(i):
    
    u = np.concatenate(([0],U[0:N+1,i],[0]))
    plt.plot(x,u)
    myAnimation.set_data(x, u)
    return myAnimation,

anim = animation.FuncAnimation(fig7,animate,frames=range(1,numbt),blit=True,repeat=False)
#writervideo = animation.FFMpegWriter(fps=60) 
#anim.save('mysample.mp4', writer=writervideo)


"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)

"Peclet number"
P = np.abs(c*Dx/nu)
print("Pe number Pe=%g\n" % P);

"CFL number"
CFL = np.abs(c*delta_t/Dx)
print("CFL number CFL=%g\n" % CFL);

#_________________________________________________________________________
"Flow parameters"
nu = 0.01
c = 2

"Scheme parameters"
beta = 1

"Number of points"

N = 64
Dx = 1/N
x = np.linspace(0,1,N+1)

order = 1

"Time parameters"
delta_t = 1/20
time = np.arange(0,3+delta_t,delta_t)
numbt = len(time)

"Initialize U here"
U = np.zeros((N-1,numbt))

for it in range(numbt-1):
    "Diffusion term"
    Diff = nu*(1/Dx**2)*(2*np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1) - np.diag(np.ones(N-2),1))

    #here, i will take note of the if condition for the advection term
    if order<2:
        cp = np.max([c,0]) #c plus
        cm = np.min([c,0]) #c minus
        "Advection term: first order upwind"
        Advp = cp*(np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1)) # A advective plus - Advp
        Advm = cm*(np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),1))  # A advective minus - Advm
    else:
        "Advection term: second order centered differences"
        Advp = -0.5*c*np.diag(np.ones(N-2),-1)
        Advm = -0.5*c*np.diag(np.ones(N-2),1)
    
    Adv = (1/Dx)*(Advp-Advm)
    A = Diff + Adv
    
    "Source term"
    F = np.ones(N-1)
    
    "Temporal terms"
    U0 = U[:,it]
    A = A + (1/delta_t)*np.diag(np.ones(N-1))
    F = F + U0/delta_t
    
    "Solution of the linear system AU=F"
    u = np.linalg.solve(A,F)
    U[:,it+1] = u
    
u = np.concatenate(([0],u,[0]))

ua = (1/c)*(x-((1-np.exp(c*x/nu))/(1-np.exp(c/nu))))

"Animation of the results"
fig8 = plt.figure()
ax = plt.axes(xlim =(0, 1),ylim =(0,1/c)) 
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
myAnimation, = ax.plot([], [],':ob',linewidth=2)
plt.grid()
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

def animate(i):
    
    u = np.concatenate(([0],U[0:N+1,i],[0]))
    plt.plot(x,u)
    myAnimation.set_data(x, u)
    return myAnimation,

anim = animation.FuncAnimation(fig8,animate,frames=range(1,numbt),blit=True,repeat=False)
#writervideo = animation.FFMpegWriter(fps=60) 
#anim.save('mysample.mp4', writer=writervideo)


"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)

"Peclet number"
P = np.abs(c*Dx/nu)
print("Pe number Pe=%g\n" % P);

"CFL number"
CFL = np.abs(c*delta_t/Dx)
print("CFL number CFL=%g\n" % CFL);

#_______________________________________________________________
