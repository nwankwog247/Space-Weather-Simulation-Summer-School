#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 17:28:58 2022

@author: nwankwog
"""

#_________________________________________________________________
"Begining of Afternoon code for Heat Equation"
#_________________________________________________________________

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation    #We have to load this
from math import pi
#%matplotlib qt
plt.close()

"Number of points"
N = 8
Dx = 1/N
x = np.linspace(0,1,N+1)
xx = np.linspace(0,1+Dx,N+2)

"Time parameters"
delta_t = 1/24
time = np.linspace(0,3+delta_t,delta_t)
numbt = len(time)

#here, i will have to specify the order of approximation
order = 2

if order<2:
    U = np.zeros(N+1, numbt)
else:
    U = np.zeros(N+2, numbt)
    
for it in range(numbt):
    if order<2:
        A = (1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))
        F = 2*(2*x**2 + 5*x -2)*np.exp(x)
        "Temporal term"
        A = A + (1/delta_t)*np.diag(np.ones(N+1))
        F = F + U[:,it]/delta_t
        
        A[0,:] = np.concatenate(([1],np.zeros(N)))
        F[0] = 0
        
        A[N,:] = (1/(2*Dx))*(np.concatenate((np.zeros(N-1),[-1,1])))
        F[N] = 0
    else:
        A = (1/Dx**2)*(2*np.diag(np.ones(N+2)) - np.diag(np.ones(N+1),-1) - np.diag(np.ones(N+1),1))
        F = 2*(2*xx**2 + 5*xx -2)*np.exp(xx)
        "Temporal term"
        A = A + (1/delta_t)*np.diag(np.ones(N+2))
        F = F + U[:,it]/delta_t
        
        A[0,:] = np.concatenate(([1],np.zeros(N+1)))
        F[0] = 0
        
        A[N+1,:] = (1/(2*Dx))*(np.concatenate((np.zeros(N-1),[-1,0,1])))
        F[N+1] = 0
        

    "Solution of the linear system AU=F"
    u = np.linalg.solve(A,F)
    U[:,it+1] = u


u = u[0:N+1]
ua = 2*x*(3-2*x)*np.exp(x)


"Plotting solution"
fig1 = plt.figure()
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
plt.legend(fontsize=12,loc='upper left')
plt.grid()
plt.axis([0, 1,0, 0.5])
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)

#____________________________________________________________________


#___________________________________________________________________________
