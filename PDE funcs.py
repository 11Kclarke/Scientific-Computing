from xml import dom
import numpy as np
import pylab as pl
from math import pi
import sympy as sp
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt

# Set problem parameters/functions
kappa = 1   # diffusion constant
L=1.0         # length of spatial domain
T=0.5         # total time to solve for
def u_I(x):
    # initial temperature distribution
    y = np.sin(pi*x/L)
    return y

def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y


## Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
"""stolen from: https://gist.github.com/cbellei/8ab3ab8551b8dfc8b081c518ccd9ada9"""
def TDMAsolver(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    nf = len(d) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
        	    
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc
    


def forwardeuler(f,T,X,boundary=0):

    mx = len(X)-1
    # Set up the solution variables
    u_j = np.zeros(X.size)      # u at current time step
    u_jp1 = np.zeros(X.size)    # u at next time step
    
    lmbda = kappa*(T[1]-T[0])/((X[1]-X[0])**2)#move inside of for loop when implementing variable timestep
    print("\n\n")
    print(lmbda)
    print((T[0]-T[1]))
    print((X[0]-X[1])**2)
    print("\n\n")
    # Set initial condition
    for i in range(mx):
        u_j[i] = u_I(X[i])
    # Boundary conditions
    u_j[0] = 0; u_j[mx] = 0
    for t in T:
        # Forward Euler timestep at inner mesh points
        # PDE discretised at position x[i], time t
        for i in range(1, mx-1):
            u_jp1[i] = u_j[i] + lmbda*(u_j[i-1] - 2*u_j[i] + u_j[i+1])
            
        # Boundary conditions
        u_jp1[0] = 0; u_jp1[mx] = 0
            
        # Save u_j at time t[j+1]
        u_j[:] = u_jp1[:]
    return u_j

def solvepde(f,t0,tn,domain,condition,deltat_max,method=forwardeuler):
    if len(condition)!= len(domain):
        raise "wrong number of innitial conditions"
    if t0 > tn and deltat_max > 0:
        deltat_max=-deltat_max
    steps = int(((tn-t0)/deltat_max))
    finalstepsize = tn-(t0+deltat_max*steps)
    print("final stepsize: ")
    print(finalstepsize)
    steps+=1
    tvals=[t0]*steps
    for i in range(steps-1):
        tvals[i+1]=deltat_max+tvals[i]
        #print(tvals[i+1])
    tvals.append(tvals[-1]+finalstepsize)
    for i in range(len(domain)):
        domain[i]=np.linspace(domain[i][0],domain[i][1],steps)
    return method()
def backwardseuler(T,X,innitial,boundary=0):
    mx = len(X)
    sol = np.zeros(shape=(T.size,X.size))
    for i in range(mx):
        sol[0][i] = innitial(X[i])
    sol[0][0]=0
    sol[0][-1]=0
    print(sol.shape)
    lmbda = kappa*(T[1]-T[0])/((X[1]-X[0])**2)
    d = np.array([2*lmbda+1]*(mx))
    d1 = np.array([-lmbda]*(mx-1))
    d2 = np.array([-lmbda]*(mx-1))
    
    for i in range(1,len(T)):
        sol[i]=TDMAsolver(d1,d,d2,sol[i-1])
        sol[i][0]=0
        sol[i][-1]=0
    return sol
def forwardeuler(T,X,innitial,boundary=0):
    mx = len(X)
    sol = np.zeros(shape=(T.size,X.size))
    for i in range(mx):
        sol[0][i] = innitial(X[i])
    sol[0][0]=0
    sol[0][-1]=0
    lmbda = kappa*(T[1]-T[0])/((X[1]-X[0])**2)
    d = np.diag([1-2*lmbda]*(mx))
    d1 = np.diag([lmbda]*(mx-1),k=1)
    d2 = np.diag([lmbda]*(mx-1),k=-1)
    A=d+d1+d2#update matrix
    
    for i in range(1,len(T)):
        sol[i]=np.matmul(A,sol[i-1])
    return sol
def CrankNicolson(T,X,innitial,boundary=0):
    mx = len(X)
    sol = np.zeros(shape=(T.size,X.size))
    for i in range(mx):
        sol[0][i] = innitial(X[i])
    sol[0][0]=0
    sol[0][-1]=0
    print(sol.shape)
    lmbda = kappa*(T[1]-T[0])/((X[1]-X[0])**2)
    d = np.diag([lmbda-1]*(mx))
    d1 = np.diag([lmbda/2]*(mx-1),k=1)
    d2 = np.diag([lmbda/2]*(mx-1),k=-1)
    A=d+d1+d2
    db = np.array([lmbda+1]*(mx))
    d1b = np.array([-lmbda/2]*(mx-1))
    d2b = np.array([-lmbda/2]*(mx-1))
    for i in range(1,len(T)):
        
        sol[i]=np.matmul(A,sol[i-1])
        sol[i]=TDMAsolver(d1b,db,d2b,sol[i])
        sol[i][0]=0
        sol[i][-1]=0
    return sol
if __name__ == "__main__":
    fig, axs = plt.subplots(2)
    """ xvals = np.linspace(0,1,200)
    tvals = np.linspace(0,1,200)
    correct =  u_exact(xvals,tvals)
    
    axs[0].plot(tvals,correct)"""
    xvals = np.linspace(0,5,10)
    tvals = np.linspace(0,1,100)
    X=backwardseuler(tvals,xvals,u_I)
    print(X.shape)
    axs[0].plot(X)
    X=CrankNicolson(tvals,xvals,u_I)
    print(X.shape)
    axs[1].plot(X)
    for i in axs:
        i.grid()
    plt.show()

    