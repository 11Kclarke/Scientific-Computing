from xml import dom
import numpy as np
import pylab as pl
from math import pi
import sympy as sp
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import animation


# Set problem parameters/functions
kappa = 5   # diffusion constant
L=10         # length of spatial domain
T=10        # total time to solve for
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
    


"""
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
"""
def backwardseuler(T,X,innitial,boundary=lambda  t : (0,0)):
    mx = len(X)
    sol = np.zeros(shape=(T.size,X.size))
    lmbda = kappa*(T[1]-T[0])/((X[1]-X[0])**2)
    for i in range(mx):
        sol[0][i] = innitial(X[i])
    sol[0][0]=sol[i][-1]+lmbda*boundary(T[i])[0]
    sol[0][-1]=sol[i][-1]+lmbda*boundary(T[i])[1]
    d = np.array([2*lmbda+1]*(mx))
    d1 = np.array([-lmbda]*(mx-1))
    d2 = np.array([-lmbda]*(mx-1))
    
    for i in range(1,len(T)):
        sol[i]=TDMAsolver(d1,d,d2,sol[i-1])
        sol[i][0]=sol[i][-1]+lmbda*boundary(T[i])[0]
        sol[i][-1]=sol[i][-1]+lmbda*boundary(T[i])[1]
    return sol
def forwardeuler(T,X,innitial,boundary=lambda  t : (0,0)):
    mx = len(X)
    sol = np.zeros(shape=(T.size,X.size))
    lmbda = kappa*(T[1]-T[0])/((X[1]-X[0])**2)
    for i in range(mx):
        sol[0][i] = innitial(X[i])
    sol[0][0]=sol[i][-1]+lmbda*boundary(T[i])[0]
    sol[0][-1]=sol[i][-1]+lmbda*boundary(T[i])[1]
    
    d = np.diag([1-2*lmbda]*(mx))
    d1 = np.diag([lmbda]*(mx-1),k=1)
    d2 = np.diag([lmbda]*(mx-1),k=-1)
    A=d+d1+d2#update matrix
    
    for i in range(1,len(T)):
        sol[i]=np.matmul(A,sol[i-1])
        sol[i][0]=sol[i][-1]+lmbda*boundary(T[i])[0]
        sol[i][-1]=sol[i][-1]+lmbda*boundary(T[i])[1]
    return sol
def CrankNicolson(T,X,innitial,boundary=lambda  t : (0,0)):
    mx = len(X)
    lmbda = kappa*(T[1]-T[0])/((X[1]-X[0])**2)
    sol = np.zeros(shape=(T.size,X.size))
    for i in range(mx):
        sol[0][i] = innitial(X[i])
    sol[0][0]=sol[i][-1]+lmbda*boundary(T[i])[0]
    sol[0][-1]=sol[i][-1]+lmbda*boundary(T[i])[1]

    d = np.diag([1-lmbda]*(mx))
    d1 = np.diag([lmbda/2]*(mx-1),k=1)#diagonals of A
    d2 = np.diag([lmbda/2]*(mx-1),k=-1)
    A=d+d1+d2
    
    db = np.array([lmbda+1]*(mx))
    d1b = np.array([-lmbda/2]*(mx-1))#diagnoals of B
    d2b = np.array([-lmbda/2]*(mx-1))
    for i in range(1,len(T)):
        
        sol[i]=np.matmul(A,sol[i-1])
        sol[i]=TDMAsolver(d1b,db,d2b,sol[i])
        sol[i][0]=sol[i][-1]+lmbda*boundary(T[i])[0]
        sol[i][-1]=sol[i][-1]+lmbda*boundary(T[i])[1]
        
    return sol
def animatepde(X,save=False,path=None):
    
    maxtemp=max(X.flatten())
    mintemp=min(X.flatten())
    centre = (maxtemp+mintemp)/2
    fig = plt.figure()
    sns.heatmap(np.zeros((10, 1)),square=False,center=centre,vmax=maxtemp,vmin=mintemp) 
    def init():
      plt.clf()
      sns.heatmap(np.zeros((10, 1)),square=False,center=centre,vmax=maxtemp,vmin=mintemp)

    def animate(i):
        D = np.asarray(X[:][i]).reshape(len(X[:][i]),1)
        plt.clf()
        sns.heatmap(D,square=False,center=centre,vmax=maxtemp,vmin=mintemp)
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=X.shape[1], repeat = False)
    if save:
        import os
        face_id = 0
        if not path:
            path=os.getcwd()
        while os.path.exists(path+"\\PDE_Anim"+str(face_id)+".gif"):
            face_id+=1
        f=path+"\\PDE_Anim"+str(face_id)+".gif"
        print(f)
        anim.save(f, writer='imagemagick',fps=15,progress_callback=lambda i, n: print(i))
    plt.show()
if __name__ == "__main__":
    xvals = np.linspace(0,L,80)
    tvals = np.linspace(0,T,200)
    """
    fig, axs = plt.subplots(4)
    
    correct = []
    for i in xvals:
        correct.append(u_exact(i,tvals))
    correct=np.array(correct).T
    
    print(correct.shape)
    axs[3].plot(correct)

    X=forwardeuler(tvals,xvals,u_I)
    print(X.shape)
    axs[0].plot(X)
    X=backwardseuler(tvals,xvals,u_I)
    print(X.shape)
    axs[1].plot(X)
    """
    X=CrankNicolson(tvals,xvals,u_I)
    print(X.shape)
    #axs[2].plot(X)
    animatepde(X,save=False)
    #for i in axs:
        #i.grid()
    plt.show()

    