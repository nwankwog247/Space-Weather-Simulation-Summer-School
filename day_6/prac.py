#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 16:00:08 2022

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


#____________________________________________________________________
"First Order Approximation"
#_____________________________________________________________________
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

"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)

#________________________________________________

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
ua = -x*(x-1)*np.exp(x)+1

"Plotting solution"
fig5 = plt.figure()
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
plt.legend(fontsize=12,loc='upper left')
plt.grid()
plt.axis([0, 1,1, 1.5])
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)

#________________________________________________________
"Number of points"
N = 8
Dx = 1/N
x = np.linspace(0,1,N+1)

"System matrix and RHS term"
#A = (1/Dx**2)*(2*np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1) - np.diag(np.ones(N-2),1))
#F = (3*x[1:N] + x[1:N]**2)*np.exp(x[1:N])
A = (1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))
#F = (3*x + x**2)*np.exp(x) #treating all point as interior points
F = 2*(2*x**2 + 5*x - 2)*np.exp(x)

#here, i will take note of the boundary condition at x=0
A[0,:] = np.concatenate(([1], np.zeros(N)))
F[0] = 0

#here, i will take note of the boundary condition at x=1

A[N,:] = (1/Dx)*np.concatenate((np.zeros(N-1),[-1,1]))
F[N] = 0

"Solution of the linear system AU=F"
U = np.linalg.solve(A,F)
u = U
ua = 2*x*(3-2*x)*np.exp(x)

"Plotting solution"
fig6 = plt.figure()
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
plt.legend(fontsize=12,loc='upper left')
plt.grid()
plt.axis([0, 1,0, 6])
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)

#__________________________________________________________________
"Number of points"
N = 16
Dx = 1/N
x = np.linspace(0,1,N+1)

"System matrix and RHS term"
#A = (1/Dx**2)*(2*np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1) - np.diag(np.ones(N-2),1))
#F = (3*x[1:N] + x[1:N]**2)*np.exp(x[1:N])
A = (1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))
#F = (3*x + x**2)*np.exp(x) #treating all point as interior points
F = 2*(2*x**2 + 5*x - 2)*np.exp(x)

#here, i will take note of the boundary condition at x=0
A[0,:] = np.concatenate(([1], np.zeros(N)))
F[0] = 0

#here, i will take note of the boundary condition at x=1

A[N,:] = (1/Dx)*np.concatenate((np.zeros(N-1),[-1,1]))
F[N] = 0

"Solution of the linear system AU=F"
U = np.linalg.solve(A,F)
u = U
ua = 2*x*(3-2*x)*np.exp(x)

"Plotting solution"
fig7 = plt.figure()
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
plt.legend(fontsize=12,loc='upper left')
plt.grid()
plt.axis([0, 1,0, 6])
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

#___________________________________________________________________
"Number of points"
N = 32
Dx = 1/N
x = np.linspace(0,1,N+1)

"System matrix and RHS term"
#A = (1/Dx**2)*(2*np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1) - np.diag(np.ones(N-2),1))
#F = (3*x[1:N] + x[1:N]**2)*np.exp(x[1:N])
A = (1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))
#F = (3*x + x**2)*np.exp(x) #treating all point as interior points
F = 2*(2*x**2 + 5*x - 2)*np.exp(x)

#here, i will take note of the boundary condition at x=0
A[0,:] = np.concatenate(([1], np.zeros(N)))
F[0] = 0

#here, i will take note of the boundary condition at x=1

A[N,:] = (1/Dx)*np.concatenate((np.zeros(N-1),[-1,1]))
F[N] = 0

"Solution of the linear system AU=F"
U = np.linalg.solve(A,F)
u = U
ua = 2*x*(3-2*x)*np.exp(x)

"Plotting solution"
fig8 = plt.figure()
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
plt.legend(fontsize=12,loc='upper left')
plt.grid()
plt.axis([0, 1,0, 6])
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)

#____________________________________________________________________
"Second Order Approximation"
#_____________________________________________________________________
"Number of points"
N = 8
Dx = 1/N
x = np.linspace(0,1,N+1)
order = 2

"System matrix and RHS term"
#A = (1/Dx**2)*(2*np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1) - np.diag(np.ones(N-2),1))
#F = (3*x[1:N] + x[1:N]**2)*np.exp(x[1:N])
A = (1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))
#F = (3*x + x**2)*np.exp(x) #treating all point as interior points
F = 2*(2*x**2 + 5*x - 2)*np.exp(x)

#here, i will take note of the boundary condition at x=0
A[0,:] = np.concatenate(([1], np.zeros(N)))
F[0] = 0

#here, i will take note of the boundary condition at x=1
if order==1:
    A[N,:] = (1/Dx)*np.concatenate((np.zeros(N-1),[-1,1]))
    F[0] = 0
else:  
    A[N,:] = (1/Dx)*np.concatenate((np.zeros(N-2),[1/2, -2, 3/2]))
    F[N] = 0

"Solution of the linear system AU=F"
U = np.linalg.solve(A,F)
u = U
ua = 2*x*(3-2*x)*np.exp(x)

"Plotting solution"
fig9 = plt.figure()
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
plt.legend(fontsize=12,loc='upper left')
plt.grid()
plt.axis([0, 1,0, 6])
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)

#____________________________________________________________________
"Number of points"
#applying a centered differences approximation
N = 8
Dx = 1/N
x = np.linspace(0,1,N+1)
xx = np.linspace(0,1+Dx,N+2)
order = 2

"System matrix and RHS term"
#here, i will take note of the boundary condition at x=1
if order<2:
    A = (1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))
    F = 2*(2*x**2 + 5*x -2)*np.exp(x)
    A[0,:] = np.concatenate(([1],np.zeros(N)))
    F[0] = 0
    A[N,:] = (1/(2*Dx))*(np.concatenate((np.zeros(N-1),[-1,1])))
    F[N] = 0
else:  
    A = (1/Dx**2)*(2*np.diag(np.ones(N+2)) - np.diag(np.ones(N+1),-1) - np.diag(np.ones(N+1),1))
    F = 2*(2*xx**2 + 5*xx -2)*np.exp(xx)
    A[0,:] = np.concatenate(([1],np.zeros(N+1)))
    F[0] = 0
    A[N+1,:] = (1/(2*Dx))*(np.concatenate((np.zeros(N-1),[-1,0,1])))
    F[N+1] = 0


"Solution of the linear system AU=F"
U = np.linalg.solve(A,F)
u = U[0: N+2]
ua = 2*xx*(3-2*xx)*np.exp(xx)

"Plotting solution"
fig10 = plt.figure()
plt.plot(xx,ua,'-r',linewidth=2,label='$u_a$')
plt.plot(xx,u,':ob',linewidth=2,label='$\widehat{u}$')
plt.legend(fontsize=12,loc='upper left')
plt.grid()
plt.axis([0, 1,0, 6])
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)

#___________________________________________________________________________
