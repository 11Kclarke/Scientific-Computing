def euler_solve(f,y0,tvals):
    steps = len(tvals)
    sol=np.zeros(steps) 
    sol[0]=y0
    for i in range(steps):
        sol[i+1]=sol[i]+euler_step(f,tvals[i+1],tvals[i]-tvals[i+1])
    
    return sol

def rk4(f,tvals,y0):
    steps = len(tvals)
    sol = np.zeros(steps)
    sol[0]=y0
    for i in range(steps):        
        sol[i+1]=rk4step(f,tvals[i],tvals[i]-tvals[i+1])+sol[i]
    return sol
"""
    if not isfunction(f):#attempts to parse function if input isnt already func
     try:
            if len(f)>1  and len(x0)==len(f):
                sols=[]
                for i in range(len(f)):
                    sols.append(Solve_to(f[i],x0[i],t0,tn))
                return sols
        #except:
                #raise "f is not function, iterable of function, or iterable of string represntation of function" 
    """
def dx_dt(t, X):
    X_dot = np.array([X[1]-t, -X[0]+t])
    return X_dot
def dx_dt2(t, X, a=0.2, b=0.1, d=0.4):
    x = X[0]
    y = X[1]

    dxdt = x*(1-x) - a*x*y/(d+x)
    dydt = b *y *(1 - y/x)

    dXdt = np.array([dxdt, dydt])
    return dXdt

def backwardseulertest(T,X,innitial,lmbdax,boundary=lambda  X,t : 1,Boundarytype="dir"):
    
    [T,sol,lmbda,shape,sqrt,X,dims]=setuppde(T,X,innitial)
    L=shape[0]
    lmbda=lmbdax
    
    print(dims)
    A=[]#array of 2d deriv matrices
    k=1
    print(np.shape(X))
    for i in range(dims):#creates deriv matrix for each dim  
        print("in deriv matrix loop")
        print(i)
        #mx num of total vals in time window
        #assuming square matrix of side length a 
        #mx = a^2 thus sqrt(2mx) = vals in diag
        print(k)
        k=L**i
        print(k)
        
        """d = np.diag([1+2*lmbda]*(sqrt))
        d1 = np.diag([-lmbda]*(sqrt-k),k=k)
        d2 = np.diag([-lmbda]*(sqrt-k),k=-k)"""
        d = np.diag([1+2*lmbda]*(sqrt))
        d1 = np.diag([-lmbda]*(sqrt-k),k=k)
        d2 = np.diag([-lmbda]*(sqrt-k),k=-k)
        M=d+d1+d2
        

        A.append(np.array(M))
    det=np.linalg.det(A[0])
    """print(L)
    print(np.shape(A[0]))
    print(np.shape(A[1]))
    print("wanted det:")
    print(det)
    for i in range(1,len(A)):
        print(i)
        print("det of matrix is prenorm :")
        
        det2=np.linalg.det(A[i])
        print(det2)
        
        scalefactor = (det/det2)**(1/(L**2))
        A[i]=A[i]*scalefactor
        print("det of matrix is postnorm :")
        print(np.linalg.det(A[i]))
    print("fin deriv matrices")
    print(len(A))"""
    diffs1=[]
    diffs2=[]
    diffs3=[]
    #A[0][0,:] = 0
    sol[0] = applycond(T[0],sol[0],boundary,X,Boundarytype,lmbda)
    for i in range(1,len(T)):
        first=sol[i][0]
        middle = sol[i][L-1]
        last=sol[i][-1]
        bleft = sol[i][-L]
        sol[i]= applycond(T[i],sol[i],boundary,X,Boundarytype,lmbda)
        print("lamda")
        print(lmbda)
        print()
        z=0
        """for M in A:
            
            if z==1:
                temp=
                temp=temp.T.flatten()
                temp=(1/2)*linalg.solve(A[0], temp, assume_a='sym')
                #temp=np.reshape(temp,(L,L)).T.flatten()
                sol[i] += temp#takes advantage of matrices always being symetric
            z+=1
            sol[i] += (1/2)*linalg.solve(A[0], sol[i-1], assume_a='sym')#takes advantage of matrices always being symetric"""
        """first= sol[0]
        last = sol[-1]"""
        
        first=sol[i][0]
        middle = sol[i][L-1]
        last=sol[i][-1]
        bleft = sol[i][-L]
        sol[i]+=(1/dims)*linalg.solve(A[0], np.reshape(sol[i-1],(L,L)).T.flatten(), assume_a='sym')
        
        print("\ndiff from first deriv matrix")
        fdiff= first-sol[i][0]
        print(fdiff)
        sdiff=middle-sol[i][L-1]
        print(sdiff)
        print((fdiff)-(sdiff))
        diffs3.append((fdiff)-(sdiff))
        
        """sol[0]=(first+sol[0])/2
        sol[-1]=(last+sol[-1])/2
        
        first= sol[0]
        last = sol[-1]
        """
        """first=sol[i][0]
        middle = sol[i][L-1]
        last=sol[i][-1]
        bleft = sol[i][-L]"""
        
        #sol[i][L-1]=middle-(first-sol[i][0])
        sol[i] += (1/dims)*linalg.solve(A[0], sol[i-1], assume_a='sym')
        
      

    sol=customreshape(shape,sol)
    
    return diffs3[0]

"""
from numba import jit, f8
'''code stolen from:  https://gist.github.com/TheoChristiaanse/d168b7e57dd30342a81aa1dc4eb3e469'''
## Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
@jit(f8[:] (f8[:],f8[:],f8[:],f8[:] )) 
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
