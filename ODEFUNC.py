
import sympy as sp
import numpy as np
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
from scipy.integrate import odeint 
from inspect import isfunction
x,y,t = sp.symbols('x,y,t', real=True)

def euler_step(dxdt,t,stepsize,fill=0):
    return dxdt(t,fill)*stepsize


def euler_solve(f,y0,tvals):
    steps = len(tvals)
    sol=np.zeros(steps) 
    sol[0]=y0
    for i in range(steps):
        sol[i+1]=sol[i]+euler_step(f,tvals[i+1],tvals[i]-tvals[i+1])
    
    return sol

def rk4step(f,x,stepsize,t=0):
    h=stepsize
    k1 = h*f(x,t)
    k2 = h*f(x+h*k1/2,t+h/2)
    k3 = h*f(x+h*k2/2,t+h/2)
    k4 = h*f(x+h*k3,t+h)
    return 1/6*(k1+2*k2+2*k3+k4)
       
def rk4(f,tvals,y0):
    steps = len(tvals)
    sol = np.zeros(steps)
    sol[0]=y0
    for i in range(steps):        
        sol[i+1]=rk4step(f,tvals[i],tvals[i]-tvals[i+1])+sol[i]
    return sol

def Solve_ode(f,tvals,y0,method=euler_step):
    steps = len(tvals)
    sol = np.zeros(steps)
    sol[0]=y0
    for i in range(steps-1):
        sol[i+1]=method(f,tvals[i],tvals[i]-tvals[i+1])+sol[i]
    return sol


def Solve_to(f,x0,t0,tn,deltat_max=0.001,method=rk4step,initialvalue=True,t_innitial_condition = None):
    if not isfunction(f):#attempts to parse function if input isnt already func
        f= sp.lambdify(x,t,sp.parse_expr(f))
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
    return  (Solve_ode(f,tvals,x0,method=method),tvals)
    
        

   

if __name__ == "__main__":
    #(x+5)*(x+1)*(x+3)**2
    f=sp.lambdify([x,t],(x+1)**2)
    a=Solve_to(f,1,-2,2,initialvalue=False,t_innitial_condition=0)
    print(np.shape(a))
    (xvals,tvals)=a
    print(np.shape(xvals))
    print(np.shape(tvals))
    print(tvals[-1])
    plt.plot(tvals, xvals)
    plt.grid()
    original = f(tvals,0)

    plt.plot(tvals,original)
    plt.show()
