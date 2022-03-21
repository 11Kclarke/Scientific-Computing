
import sympy as sp
import numpy as np
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
from scipy.integrate import odeint 
from inspect import isfunction
x,y,t = sp.symbols('x,y,t', real=True)

def euler_step(dxdt,x,stepsize,t=0):
    #print(t)
    return dxdt(t,x)*stepsize


def euler_solve(f,y0,tvals):
    steps = len(tvals)
    sol=np.zeros(steps) 
    sol[0]=y0
    for i in range(steps):
        sol[i+1]=sol[i]+euler_step(f,tvals[i+1],tvals[i]-tvals[i+1])
    
    return sol

def rk4step(f,x,stepsize,t=0):
    h=stepsize
    k1 = h*f(t,x)
    k2 = h*f(t+h/2,x+h*k1/2)
    k3 = h*f(t+h/2,x+h*k2/2)
    k4 = h*f(t+h,x+h*k3)
    return 1/6*(k1+2*k2+2*k3+k4)
       
def rk4(f,tvals,y0):
    steps = len(tvals)
    sol = np.zeros(steps)
    sol[0]=y0
    for i in range(steps):        
        sol[i+1]=rk4step(f,tvals[i],tvals[i]-tvals[i+1])+sol[i]
    return sol

def Solve_ode(f,tvals,y0,method=rk4step):
    steps = len(tvals)
    print(y0)
    """
    if not isfunction(f):
        if isfunction(f[0]):
            
    """
    #print(np.shape(f([tvals[0],tvals[0]],0)))
    #tvals=np.array([tvals,tvals],dtype=float)
    #x = np.zeros(shape=(len(solve_for), initial_condition.size))
    


    sol = np.zeros(shape=(len(y0),len(tvals))).T
    print(np.shape(sol))
    print(np.shape(sol[0]))
    sol[0]=y0
    print(sol[0])
    #print(f)
    for i in range(steps-1):
        sol[i+1]=method(f,sol[i],tvals[i]-tvals[i+1],t=tvals[i])+sol[i]
    return sol


def Solve_to(f,x0,t0,tn,deltat_max=0.001,method=rk4step,initialvalue = True ,t_innitial_condition = None):
    if type(f)==str:
        f= sp.lambdify([x,t],sp.parse_expr(f))
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
    if abs(deltat_max)>abs(t0-tn):
        raise "max step size larger than t range"
    steps = abs(int((t0-tn)/deltat_max))
    finalstepsize = tn-(t0+deltat_max*steps)
    tvals=[t0]*steps
    if not initialvalue:
        (forwardx, forwardt)= Solve_to(f,x0,t_innitial_condition,tn)
        (backwardsx, backwardst) = Solve_to(f,x0,t_innitial_condition,t0,deltat_max=-deltat_max)
        tvals = np.append(np.flip(backwardst),forwardt)
        xvals = np.append(np.flip(backwardsx),forwardx)
        return (xvals,tvals)
    for i in range(steps-1):
        tvals[i+1]=deltat_max+tvals[i]
    tvals.append(tvals[-1]+finalstepsize)
    #tvals = np.vstack((tvals,tvals)).T
    print(np.shape(tvals))
    #tvals= [tvals]*len(x0)
    return  (Solve_ode(f,tvals,x0,method=method),tvals)
    
def dx_dt(t, X):
    
    X_dot = np.array([X[1], -X[0]])
    return X_dot



def dx_dt2(t, X, a=0.1, b=0.1, d=0.1):
    x = X[0]
    y = X[1]

    dxdt = x*(1-x) - a*x*y/(d+x)
    dydt = b *y *(1 - y/x)

    dXdt = np.array([dxdt, dydt])
    return dXdt

  

if __name__ == "__main__":
    #(x+5)*(x+1)*(x+3)**2
    f=sp.lambdify([x,t],(x+1)**2)
    h=sp.lambdify([x,t],t)
    
    g = sp.lambdify([x,t],-x**2)
    f=[h,g]
    #dx_dt(0,)
    a=Solve_to(dx_dt2,[0,1],np.pi,3*np.pi)
    #print(np.shape(a))
    (xvals,tvals)=a
    
    print(np.shape(xvals))
    print(np.shape(tvals))
    #print(tvals[-1])
    
   
    plt.plot(xvals[:,1])

    plt.grid()
    #original = f(tvals,0)

    #plt.plot(tvals,original)
    plt.show()
