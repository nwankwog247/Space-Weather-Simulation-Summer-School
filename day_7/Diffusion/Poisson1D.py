#!/usr/bin/env python
"""
Solution of a 1D Poisson equation: -u_xx = f
Domain: [0,1]
BC: u(0) = u(1) = 0
with f = (3*x + x^2)*exp(x)

Analytical solution: -x*(x-1)*exp(x)

Finite differences (FD) discretization: second-order diffusion operator
"""
__author__ = 'Jordi Vila-Pérez'
__email__ = 'jvilap@mit.edu'


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

"System matrix and RHS term"
A = (1/Dx**2)*(2*np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1) - np.diag(np.ones(N-2),1))
F = (3*x[1:N] + x[1:N]**2)*np.exp(x[1:N])

"Solution of the linear system AU=F"
U = np.linalg.solve(A,F)
u = np.concatenate(([0],U,[0]))
ua = -x*(x-1)*np.exp(x)

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


#_________________________
"Number of points"
N = 16
Dx = 1/N
x = np.linspace(0,1,N+1)

"System matrix and RHS term"
A = (1/Dx**2)*(2*np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1) - np.diag(np.ones(N-2),1))
F = (3*x[1:N] + x[1:N]**2)*np.exp(x[1:N])

"Solution of the linear system AU=F"
U = np.linalg.solve(A,F)
u = np.concatenate(([0],U,[0]))
ua = -x*(x-1)*np.exp(x)

"Plotting solution"
fig2 = plt.figure()
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


#__________________________________________
"Number of points"
N = 32
Dx = 1/N
x = np.linspace(0,1,N+1)

"System matrix and RHS term"
A = (1/Dx**2)*(2*np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1) - np.diag(np.ones(N-2),1))
F = (3*x[1:N] + x[1:N]**2)*np.exp(x[1:N])

"Solution of the linear system AU=F"
U = np.linalg.solve(A,F)
u = np.concatenate(([0],U,[0]))
ua = -x*(x-1)*np.exp(x)

"Plotting solution"
fig3 = plt.figure()
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

#____________________________________________________________
"Number of points"
N = 8
Dx = 1/N
x = np.linspace(0,1,N+1)

"System matrix and RHS term"
#A = (1/Dx**2)*(2*np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1) - np.diag(np.ones(N-2),1))
#F = (3*x[1:N] + x[1:N]**2)*np.exp(x[1:N])
A = (1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))
F = (3*x + x**2)*np.exp(x) #treating all point as interior points

"Solution of the linear system AU=F"
U = np.linalg.solve(A,F)
u = np.concatenate(([0],U,[0]))
ua = -x*(x-1)*np.exp(x)

"Plotting solution"
fig3 = plt.figure()
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

#____________________________________________________________________________
"Number of points"
N = 8
Dx = 1/N
x = np.linspace(0,1,N+1)

"System matrix and RHS term"
#A = (1/Dx**2)*(2*np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1) - np.diag(np.ones(N-2),1))
#F = (3*x[1:N] + x[1:N]**2)*np.exp(x[1:N])
A = (1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))
F = (3*x + x**2)*np.exp(x) #treating all point as interior points

#here, i will take note of the boundary condition at x=0
A[0,:] = np.concatenate(([1], np.zeros(N)))
F[0] = 0

#here, i will take note of the boundary condition at x=1
A[N,:] = np.concatenate((np.zeros(N),[1]))
F[N] = 0
"Solution of the linear system AU=F"
U = np.linalg.solve(A,F)
u = U
ua = -x*(x-1)*np.exp(x)

"Plotting solution"
fig4 = plt.figure()
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
plt.legend(fontsize=12,loc='upper left')
plt.grid()
plt.axis([0, 1,0, 0.5])
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

#______________________________________________
"Number of points"
N = 8
Dx = 1/N
x = np.linspace(0,1,N+1)

"System matrix and RHS term"
#A = (1/Dx**2)*(2*np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1) - np.diag(np.ones(N-2),1))
#F = (3*x[1:N] + x[1:N]**2)*np.exp(x[1:N])
A = (1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))
F = (3*x + x**2)*np.exp(x) #treating all point as interior points

#here, i will take note of the boundary condition at x=0
A[0,:] = np.concatenate(([1], np.zeros(N)))
F[0] = 1

#here, i will take note of the boundary condition at x=1
A[N,:] = np.concatenate((np.zeros(N),[1]))
F[N] = 1
"Solution of the linear system AU=F"
U = np.linalg.solve(A,F)
u = U
ua = -x*(x-1)*np.exp(x)

"Plotting solution"
fig5 = plt.figure()
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
plt.legend(fontsize=12,loc='upper left')
plt.grid()
plt.axis([0, 1,1, 1.5])
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)








