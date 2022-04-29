from decimal import setcontext
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
from scipy import sparse
import sys

np.set_printoptions(threshold=np.inf)
# Set problem parameters/functions
kappa = 10  # diffusion constant
L=10       # length of spatial domain
T=10      # total time to solve for
def u_I(x):
    # initial temperature distribution
    #y = np.sin(pi*x/L)
    y = np.sin(pi*x/L)*10
    return y

def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

def u_I2d(x):
    # initial temperature distribution
    
    return (2*np.sin(np.pi*x[1]/4*L)+2*np.cos(np.pi*x[0]/4*L))
    #if x[0]==0 or x[0]==L:
        #return 5
    return 0


"""
def u_I2d(x):
    # initial temperature distribution
    
    #return 4*(np.sin(x[1])**2+np.cos(x[0])**2)
    
    return 0 
    """
 

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
    
def customreshape(shape,sol):
    
    solshaped=[]
    L=int(len(sol[0])**(1/2))
    if len(shape)>1:
        for i in range(1,len(sol)-2):
            #i flattened list of spatial stuff at time i
            soli=sol[i].reshape(L,L)
            #solshaped.append(soli)
            solshaped.append(soli[2:-2,2:-2])
        sol=np.array(solshaped)
    return sol



def setuppde(T,X,innitial):
    print("\npre proccessing started\n")
    X=np.array(X)
    #X should be A grid of tuples where the grid is dimensionality of the spatial domain and tuple length of dim of spatial domain
    shape=np.shape(X)#ignores the last dim, its the size of the tuples
    #number of spatial values in 1 time window 1 spatial value is a tuple
    #print(X)
    
    if len(shape)>1:
        #has been passed tuples
        dims=X.ndim-1
        mx = int(np.prod(shape[:-1]))
        print(mx)
        
        sideL=int(mx**(1/2))
        print(sideL)
        X=X.flatten().reshape(mx,shape[-1])
        #print(X)
    else:
        #has been passed individual values
        mx=len(X)
        dims=1
    sol = np.zeros(shape=(T.size,mx))
    for i in range(mx):
        sol[0][i] = innitial(X[i])# innital condition function should take tuple of length dim of spatial domain and return 1 val
    temp=X
    
    X=X.flatten()
    i=0
    while (X[i+1]-X[i] == 0) or (T[i+1]-T[i] == 0): 
        i+=1
    lmbda = kappa*(T[i+1]-T[i])/((2*dims*(X[i+1]-X[i]))**2)
    X=temp
    if dims==1:
        sqrt=mx
    else:
        sqrt = int((2*mx)**1/2)
    print("\npre proccessing done\n")
    
    return [T,sol,lmbda,shape,sqrt,X,dims]


def forwardeuler(T,X,innitial,boundary=lambda  x,t : 1,Boundarytype=["dir"]):#2d and 1d but 2d really unstable


    [T,sol,lmbda,shape,sqrt,X,dims]=setuppde(T,X,innitial)
    print(sol[0])
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
        
        d = np.diag([1-2*lmbda]*int(sqrt))
        d1 = np.diag([lmbda]*(int(sqrt)-k),k=k)
        d2 = np.diag([lmbda]*(int(sqrt)-k),k=-k)
        M=d+d1+d2
        k*=shape[:-1][i]
        
        A.append(M)

    """if each row is n long, the under and over element will be at -+n
    if there are 3 spatial dimensions the element infront/behind in 3rd dim will be +-(n*m) where m is length of 3rd dim
    as n*m must be stepped over before beingback in place in first dim 2 dims and same place in 3rd
    idk if this will work for 3d cant visualize it or check results so dont even know if i will test it
    however it makes sense that it would work"""
    print(T)
    for i in range(1,len(T)):
        print(i)
        for M in A:
            
            sol[i]=np.matmul(sol[i-1].flatten(),M)
            sol[i]= applycond(T[i],sol[i],boundary,X,Boundarytype,lmbda)
       
    #sol currently flattened spacial dims for each t
    sol=customreshape(shape,sol)
    return sol

def CrankNicolson(T,X,innitial,boundary=lambda  x,t : 1,Boundarytype=["dir+"]):#1d only
    mx = len(X)
    lmbda = kappa*(T[1]-T[0])/((X[1]-X[0])**2)
    sol = np.zeros(shape=(T.size,X.size))
    for i in range(mx):
        sol[0][i] = innitial(X[i])
    
    #sol[0][0]=sol[i][-1]+lmbda*boundary(T[i])[0]
    #sol[0][-1]=sol[i][-1]+lmbda*boundary(T[i])[1]

    d = np.diag([1-lmbda]*(mx))
    d1 = np.diag([lmbda/2]*(mx-1),k=1)#diagonals of A
    d2 = np.diag([lmbda/2]*(mx-1),k=-1)
    A=d+d1+d2
    
    db = np.array([lmbda+1]*(mx))
    d1b = np.array([-lmbda/2]*(mx-1))#diagnoals of B
    d2b = np.array([-lmbda/2]*(mx-1))
    for i in range(1,len(T)):
        sol[i-1]= applycond(T[i],sol[i-1],boundary,X,Boundarytype,lmbda)
        sol[i]=np.matmul(A,sol[i-1])
        sol[i]=TDMAsolver(d1b,db,d2b,sol[i-1])
        #sol[i][0]=sol[i][-1]+lmbda*boundary(T[i])[0]
        #sol[i][-1]=sol[i][-1]+lmbda*boundary(T[i])[1]
        
    return sol
    
def backwardseuler(T,X,innitial,boundary=lambda  X,t : 1,Boundarytype=["dir+"]):#2d 4 way symetric, or 1d 
    
    [T,sol,lmbda,shape,sqrt,X,dims]=setuppde(T,X,innitial)
    L=shape[0]
    print(dims)
    
    k=1
    side=np.array([-lmbda]*(sqrt-k))
    d = np.array([1+2*lmbda]*(sqrt))
    
    #M=d1+d2+d
    #sol[0]= applycond(T[0],sol[0],boundary,X,Boundarytype,lmbda)
    if dims==1:

        for i in range(1,len(T)):
            sol[i-1]= applycond(T[i],sol[i-1],boundary,X,Boundarytype,lmbda)   
            #sol[i]+=(linalg.solve(M, sol[i-1], assume_a='sym'))
            sol[i]= TDMAsolver(side,d,side,sol[i-1])
    else:  
        for i in range(1,len(T)):
            print(i)
            sol[i]= applycond(T[i],sol[i-1],boundary,X,Boundarytype,lmbda)
            soli=np.reshape(sol[i],(L,L))
            lb=sol[i][0:L]
            rb=sol[i][-L:]
            hb=sol[i][0:L]
            lowerb=sol[i][-L:]
            
            
            #ydiff=(1/dims)*linalg.solve(M, np.reshape(sol[i-1],(L,L)).T.flatten(), assume_a='sym')
            xdiff = TDMAsolver(side,d,side,sol[i])
            ydiff = TDMAsolver(side,d,side,soli.T.flatten())
            xdiff[0:L]=lb
            xdiff[-L:]=rb
            ydiff[0:L]=hb
            ydiff[-L:]=lowerb
            diff= np.reshape(ydiff,(L,L)).T.flatten()+xdiff
            diff=xdiff+ydiff
            diff=diff/(dims)
            sol[i]=diff
            
    sol=customreshape(shape,sol)
    return sol


    
def ADI(T,X,innitial,boundary=lambda  X,t : 1,Boundarytype=["dir+"]):#2d only
    
    [T,sol,lmbda,shape,sqrt,X,dims]=setuppde(T,X,innitial)
    dx=X[0][0]-X[1][0]
    L=shape[0]
    #sqrt number of values on 1 edge of square domain
    #deriv matrix stuff
    #sqrt+=1
    #magic numbers from "inspiration code"
    """
    d = -(2+(dx**2)/(beta*dt))
    B = (2*((dx/dy)**2)-((dx**2)/(beta*dt)))
    e = -(2+(dy**2)/(beta*dt))
    C = (2*((dy/dx)**2)-((dy**2)/(beta*dt)))
    im assuming square so
    dy/dx=dx/dy=1
    beta = kappa?
    (dx**2)/(beta*dt)=1/lmbda?
    d=-(2+1/lmbda)
    B=2-1/lmbda
    e=d
    C=B
    """
    magicnumber=-(2+1/lmbda)#explained in detail in writeup
    magicnumber2=2-1/lmbda
    d=np.ones(L) * magicnumber 
    side= np.ones(L-1)
    tsteps=len(T)-1
    steplist=list(range(1,L-1))
    #tsteps= int(len(T)/2)#each half step is step
    #sol[0]=applycond(T[0],sol[0],boundary,X,Boundarytype,lmbda)
    
    for t in range(tsteps):
        sol[t]=applycond(T[t],sol[t],boundary,X,Boundarytype,lmbda)#imperative conditions are applied before
        sol[t]=applycond(T[t],sol[t],boundary,X,Boundarytype,lmbda)#as adi does 2 steps in one certain boundary stuff needs to be double applied
        print(t)
        tsol=np.reshape(sol[t],(L,L))
        #loops kept seperate for ease of later expansion
        for i in steplist:#Xloop
            xtarr=tsol[i,:]*magicnumber2-(tsol[i-1,:]+tsol[i+1,:])
            #xtarr[0]-=tsol[i,0]
            #xtarr[-1]-=tsol[i,-1]
            b1=tsol[i,0]
            b2=tsol[i,-1]
            tsol[i,:]=TDMAsolver(side,d,side,xtarr)
            tsol[i,-1]=b2
            tsol[i,0]=b1
        
        for j in steplist:#Yloop#reversed this loops direction
            ytarr=tsol[:,j]*magicnumber2-(tsol[:,j-1]+tsol[:,j+1])
            #ytarr[0]-=tsol[0,j]
            #ytarr[-1]-=tsol[-1,j]
            b1=tsol[0,j]
            b2=tsol[-1,j]
            tsol[:,j]=TDMAsolver(side,d,side,ytarr)
            tsol[0,j]=b1
            tsol[-1,j]=b2
        sol[t+1]=tsol.flatten()
        steplist=np.flip(steplist)    
    sol=customreshape(shape,sol)
    print(L)
    print(np.shape(sol))
    return sol


def transposer(A,L):
    A=np.reshape(A,(L,L)).T.flatten()


def applycond(t,sol,boundary,X,boundarytype,lmda):
    
    #applies a function of x and t to either edges of sol, or entire
    """A function used for boundary conditions or heat source should output
     1 temperature for 1 set of coordinates, and time. 
     With the coordinates given as a tuple, as the first positional argument,
      and the time given as second positional argument."""
    dims= X.ndim
    sidelength = int(len(sol)**(1/dims))
    if dims==1:
        sidelength=1
    sol=sol.flatten()
    #2boundaries per dim
    """
    X and sol should both be flat and have same num of elements, where each element of X is tuple
    assuming square domain, n=sidelength= int(len(sol)**1/2), first and last n values boundary, 
    as well as every nth value, if cube domain every n*n value also edge
    """
    if "dirh" in boundarytype:
        for d in range(dims):
            for i in range(0,sidelength):
                sol[i*((sidelength)**d)]=0#close boundary on axis d
                
                sol[(-i*((sidelength)**d))-1]=0#far boundary on axis d
    elif "dir+" in boundarytype:
        for d in range(dims):
            
            for i in range(0,sidelength):
                
                sol[(i)*(sidelength**d)]+=lmda*boundary(X[(i)*(sidelength**d)],t)#close boundary on axis d
                
                sol[-(i)*((sidelength)**d)-1]+=lmda*boundary(X[(-(i)*((sidelength)**d))-1],t)#far boundary on axis d
    elif "dir" in boundarytype:
        for d in range(dims):
            for i in range(0,sidelength):
                
                sol[i*((sidelength)**d)]=lmda*boundary(X[i*((sidelength)**d)],t)#close boundary on axis d
                
                sol[-i*((sidelength)**d)-1]=lmda*boundary(X[(-i*((sidelength)**d))-1],t)#far boundary on axis 
    
    if "heatsource" in boundarytype:
        print("domain wide application")
        for i in range(len(sol)):
            sol[i]+=lmda*boundary(X[i],t) 
    if "periodicy" in boundarytype or "periodicx" in boundarytype:#if periodic with no heat source use func that always gives 0, ie default
        #print("periodic")
        dimlist=[]
        
        side= np.ones(sidelength-1)
        if "periodicx" in boundarytype:
            dimlist.append(0)
        if "periodicy" in boundarytype:
            dimlist.append(1)
        for ree in range(1):    
            for d in dimlist:
                for i in range(1,sidelength-2):
                    #sol[i*(sidelength)**d]=sol[(-i*(sidelength)**d)-1]
                    #i+=2
                    
                    
                    right = sol[(i)*(sidelength)**d]
                    current = sol[-(i+1)*(sidelength)**d-1]
                    left = sol[-(i+2)*(sidelength)**d-1]
                    diff=lmda*(2*current-left-right)
                    
                    current = sol[(i)*(sidelength)**d]
                    left = sol[-(i)*(sidelength)**d-1]
                    right = sol[(i+1)*(sidelength)**d]
                    sol[(i)*(sidelength)**d]-=lmda*(2*current-left-right)
                    sol[-(i+1)*(sidelength)**d-1]-=diff
                    
                    
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
        anim.save(f, writer='imagemagick',fps=30,progress_callback=lambda i, n: print(str(i)+" / "+str(X.shape[0])+" frames "))
    plt.show()



def exampleheatsource(x,t,L=L,T=T):
    y=x[1]/L
    x=x[0]/L
    if (y>0.1 and y < 0.2) and (x>0.2 and x < 0.8):
        return -4
    else:
        return 0
        
if __name__ == "__main__":
    xvals = np.linspace(0,L,20)
    tvals = np.linspace(0,T,600)
    coords=create2dgrid(0,L,31)
    #X=backwardseuler(tvals,xvals,u_I)
    #X=forwardeuler(tvals,xvals,u_I)
    #X=CrankNicolson(tvals,xvals,u_I)
    #plt.plot(X[1])
    #plt.plot(X[0])
    #plt.show()
    X=ADI(tvals,coords,u_I2d)
    print(np.shape(X))
    temp=[]
    c=0
    n=2
    for i in X:
        c+=1
        if c%n==0:
            temp.append(i)
        if c%500==0:
            n+=1
    X=np.array(temp)
    
    animatepde(X,save=False)
    
    plt.show()

    