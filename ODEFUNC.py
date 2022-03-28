
import sympy as sp
import numpy as np
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
from scipy.integrate import odeint 
from inspect import isfunction
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import inspect
x,y,t = sp.symbols('x,y,t', real=True)

def euler_step(dxdt,x,stepsize,t):
    
    
    return dxdt(t,x)*stepsize


def euler_solve(f,y0,tvals):
    steps = len(tvals)
    sol=np.zeros(steps) 
    sol[0]=y0
    for i in range(steps):
        sol[i+1]=sol[i]+euler_step(f,tvals[i+1],tvals[i]-tvals[i+1])
    
    return sol

def rk4step(f,x,stepsize,t):
    
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

def Solve_ode(f,tvals,y0,method=euler_step,event=None):
    steps = len(tvals)
    print(y0)
    sol = np.zeros(shape=(len(y0),steps)).T  
    sol[0]=y0
    if event != None:
        sign = np.sign(event(y0,tvals[0]))
        eventT=[]
        for i in range(steps-1):
            sol[i+1]=method(f,sol[i],tvals[i+1]-tvals[i],tvals[i])+sol[i]
            if np.sign(event(sol[i+1],tvals[i])) != sign:
                sign=np.sign(event(sol[i+1],tvals[i]))#if event has sign change event must have happened
                inbetweensteps = Solve_to(f,sol[i],tvals[i],tvals[i+1],deltat_max=1*10**(-16),method=rk4step)#replace deltattmax with maximum effective accuraccy 
                #Solves ode between steps at maximum accuracy to find more exact event locations
                for i in inbetweensteps:
                    if abs(i[0])<1*10**(-15):
                        eventT.append(i[1])
        return sol,eventT    
    for i in range(steps-1):
        sol[i+1]=method(f,sol[i],tvals[i+1]-tvals[i],tvals[i])+sol[i]
    
    return sol


def Solve_to(f,x0,t0,tn,deltat_max=0.01,method=rk4step,initialvalue = True ,t_innitial_condition = None,event=None):
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
        
    
    if t0 > tn and deltat_max > 0:
        deltat_max=-deltat_max
    steps = int(((tn-t0)/deltat_max))
    
    finalstepsize = tn-(t0+deltat_max*steps)
    print("final stepsize: ")
    print(finalstepsize)
    steps+=1
    tvals=[t0]*steps
    if not initialvalue:
        print("not innitial value")
        (forwardx, forwardt)= Solve_to(f,x0,t_innitial_condition,tn)
        (backwardsx, backwardst) = Solve_to(f,x0,t_innitial_condition,t0,deltat_max=-deltat_max)
        tvals = np.append(np.flip(backwardst),forwardt)
        xvals = np.append(np.flip(backwardsx),forwardx)
        return (xvals,tvals)

    for i in range(steps-1):
        tvals[i+1]=deltat_max+tvals[i]
        #print(tvals[i+1])
    
    
    tvals.append(tvals[-1]+finalstepsize)
    
    if abs(finalstepsize)>abs(deltat_max):#remove before submit
        print("reeeeeeeeeeeeeeeeeeeee")
    print("t rounding error:")
    print(tvals[-1]-tn)


    print(np.shape(tvals))
   
    return  (Solve_ode(f,tvals,x0,method=method,event=event),tvals)
    
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


def findcycle(X,f,t0=0,period=None,phasecond=lambda  y : y[1]):
    if period == None:
        def G(x0,t0=0,phasecond=phasecond):
            tn= x0[0]
            phasecond=phasecond(x0)
            x_array = solve_ivp(f,(t0,tn),x0[1:],max_step = 0.01)
            return np.append(x0[1:]-x_array.y[:,-1],phasecond)
    else:     
        def G(x0,T=period):
        
            t_array = [0,T]
    
            x_array = solve_ivp(f,(t_array[0],t_array[1]),x0,max_step = 0.01)

            return x0[1:]-x_array.y[:,-1]
    return fsolve(G,X)

def mass_spring(t,x,const = [0.1,0.1,32,0.1,0.1]):
    m=const[0]
    gamma=const[1]
    w=const[2]
    c=const[3]
    k=const[4]
    dx1 = x[1]
    dx2 = (1/m)*(gamma*np.sin(w*t)-(c*x[1])-(k*x[0]))
    return np.array([dx1,dx2])

if __name__ == "__main__":
    b= findcycle([1,2,3],mass_spring)
    
    fig, axs = plt.subplots(2)
    
    sol = solve_ivp(mass_spring,(0,b[0]),b[1:],max_step = 0.01)
    (xvals,tvals)=Solve_to(mass_spring,b[1:],0,b[0])
    print(b)
    print(tvals[-1])
    print(sol.t[-1])
    print(xvals[-1,:])
    print(sol.y[:,0])
    axs[0].plot(sol.t,sol.y[1,:])
    axs[0].plot(sol.t,sol.y[0,:])
    axs[1].plot(xvals[:,0],xvals[:,1])
    #axs[1].plot(tvals,xvals[:,1])
    
    """
    Solve to broken for t dependent
    """
    for i in axs:
        i.grid()
    #original = f(tvals,0)

    #plt.plot(tvals,original)
    plt.show()
