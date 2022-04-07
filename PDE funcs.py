from xml import dom
import numpy as np
import pylab as pl
from math import pi
import sympy as sp
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import animation
import scipy.linalg as linalg

# Set problem parameters/functions
kappa = 16   # diffusion constant
L=8        # length of spatial domain
T=4       # total time to solve for
def u_I(x):
    # initial temperature distribution
    y = np.sin(pi*x/L)
    return y

def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

def u_I2d(x):
    # initial temperature distribution
    
    return np.sin(np.pi*x[0]/L)+np.cos(np.pi*x[1]/L)
 

def create2dgrid(min,max,steps):
    if steps==0:
        return np.array([(min,min),(max,max)])
    A=[]
    x=y=min
    stepsize=(max-x)/(steps-1)
    for j in range(steps):
        row=[]
        for i in range(steps):
            row.append((x,y))
            x+=stepsize
        x=min
        y+=stepsize
        A.append(row)
    return np.array(A)



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
def customreshape(shape,sol):
    solshaped=[]
    if len(shape)>1:
        for i in sol:
            #i flattened list of spatial stuff at time i
            solshaped.append(i.reshape(shape[:-1]))
        sol=np.array(solshaped)
    return sol
def setuppde(T,X,innitial,boundary=lambda  t : (0,0)):
    X=np.array(X)
    #X should be A grid of tuples where the grid is dimensionality of the spatial domain and tuple length of dim of spatial domain
    shape=np.shape(X)#ignores the last dim, its the size of the tuples
    #number of spatial values in 1 time window 1 spatial value is a tuple
    if len(shape)>1:
        mx = int(np.prod(shape[:-1]))
        X=X.flatten().reshape(mx,shape[-1])
    else:
        mx=len(X)
    sol = np.zeros(shape=(T.size,mx))
    for i in range(mx):
        sol[0][i] = innitial(X[i])# innital condition function should take tuple of length dim of spatial domain and return 1 val
        
    """sol[0][0]=sol[i][-1]+lmbda*boundary(T[i])[0]
    sol[0][-1]=sol[i][-1]+lmbda*boundary(T[i])[1]"""
    
    
    X=X.flatten()
    i=0
    while (X[i+1]-X[i] == 0) or (T[i+1]-T[i] == 0): 
        i+=1
    lmbda = kappa*(T[i+1]-T[i])/((X[i+1]-X[i])**2)
    sqrt = (2*mx)**1/2
    print("\npre proccessing done\n")
    return [T,X,sol,lmbda,shape,sqrt]
def backwardseuler(T,X,innitial,boundary=None):
    [T,X,sol,lmbda,shape,sqrt]=setuppde(T,X,innitial)
    
  
    """print(np.shape(sol))
    print(sol[0])
    print(sol[0][0])
    print(lmbda*boundary(T[0])[0])
"""
    k=1
    d = np.diag([1+2*lmbda]*int(sqrt))
    d1 = np.diag([-lmbda]*(int(sqrt)-k),k=k)
    d2 = np.diag([-lmbda]*(int(sqrt)-k),k=-k)
    M=d+d1+d2
    A=[M]#array of 2d deriv matrices
    for i in range(1,len(shape)-1):#creates deriv matrix for each dim  
        #mx num of total vals in time window
        #assuming square matrix of side length a 
        #mx = a^2 thus sqrt(2mx) = vals in diag
        k*=shape[:-1][i]
        d = np.diag([1+2*lmbda]*int(sqrt))
        d1 = np.diag([-lmbda]*(int(sqrt)-k),k=k)
        d2 = np.diag([-lmbda]*(int(sqrt)-k),k=-k)
        M=d+d1+d2
    #sol[0][0]=lmbda*boundary(T[0])[0]
    #sol[0][-1]=lmbda*boundary(T[0])[1]
    for i in range(1,len(T)):
        for M in A:
            #sol[i]+=np.linalg.solve(M,sol[i-1])
            sol[i] += linalg.solve(M, sol[i-1], assume_a='sym')#takes advantage of matrices always being symetric
    """for i in range(1,len(T)):
        if len(sol[0][0])==1:
            sol[i]=TDMAsolver(d1,d,d2,sol[i-1])
        else:
            A=np.diag(d)+np.diag(d1,k=1)+np.diag(d2,k=-1)
            #print(A)
            #print(np.shape(A))
            #print(np.shape(sol[i-1]))
            sol[i]= np.linalg.solve(A,sol[i-1])
        sol[i][0]=sol[i][-1]+lmbda*boundary(T[i])[0]
        sol[i][-1]=sol[i][-1]+lmbda*boundary(T[i])[1]"""
    sol=customreshape(shape,sol)
    return sol


def forwardeuler(T,X,innitial,boundary=lambda  t : (0,0)):
    [T,X,sol,lmbda,shape,sqrt]=setuppde(T,X,innitial)
    
    k=1
    d = np.diag([1-2*lmbda]*int(sqrt))
    d1 = np.diag([lmbda]*(int(sqrt)-k),k=k)
    d2 = np.diag([lmbda]*(int(sqrt)-k),k=-k)
    M=d+d1+d2
    A=[M]#array of 2d deriv matrices
    
    
    
    for i in range(1,len(shape)-1):#creates deriv matrix for each dim  
        #mx num of total vals in time window
        #assuming square matrix of side length a 
        #mx = a^2 thus sqrt(2mx) = vals in diag
        k*=shape[:-1][i]
        d = np.diag([1-2*lmbda]*int(sqrt))
        d1 = np.diag([lmbda]*(int(sqrt)-k),k=k)
        d2 = np.diag([lmbda]*(int(sqrt)-k),k=-k)
        M=d+d1+d2
        print("reeeeeeeeeeeeeeeeeeeee")
        
        A.append(M)

    """if each row is n long, the under and over element will be at -+n
    if there are 3 spatial dimensions the element infront/behind in 3rd dim will be +-(n*m) where m is length of 3rd dim
    as n*m must be stepped over before beingback in place in first dim 2 dims and same place in 3rd
    idk if this will work for 3d cant visualize it or check results so dont even know if i will test it
    however it makes sense that it would work"""
    for i in range(1,len(T)):
        for M in A:
            sol[i]+=np.matmul(sol[i-1].flatten(),M)
        #sol[i][0]=sol[i][-1]+lmbda*boundary(T[i])[0]
        #sol[i][-1]=sol[i][-1]+lmbda*boundary(T[i])[1]
    #sol currently flattened spacial dims for each t
    sol=customreshape(shape,sol)
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
    if len(np.shape(X[0]))<2:
        
        temp=[]
        for i in range(len(X)):
            #print(X[:][i])
            D = np.asarray(X[:][i]).reshape(len(X[:][i]),1)
            #print(D)
            temp.append(D)
    
        X=np.array(temp)
    
    centre = (maxtemp+mintemp)/2
    fig = plt.figure()
    sns.heatmap(X[0],square=False,center=centre,vmax=maxtemp,vmin=mintemp) 
    def init():
      plt.clf()
      sns.heatmap(X[0],square=False,center=centre,vmax=maxtemp,vmin=mintemp)

    def animate(i):
        
        #D = np.asarray(X[:][i]).reshape(len(X[:][i]),X.shape[2])
        plt.clf()
        sns.heatmap(X[i],square=False,center=centre,vmax=maxtemp,vmin=mintemp)
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=X.shape[0], repeat = False)
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
    #coords = np.linspace(0,L,16)
    tvals = np.linspace(0,T,300)
    coords=create2dgrid(0,L,24)
 
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
   
    X=backwardseuler(tvals,coords,u_I2d)
    print("innital and final val")
    print(X[0])
    print("\n\n\n")
    print(X[-1])
    print("\n\n\n")
    print(X)
    print("\n\n\n")
    print(X.shape)
    #plt.plot(X[1])
    #axs[2].plot(X)

    animatepde(X,save=True)
    #for i in axs:
        #i.grid()
    plt.show()

    