
from more_itertools import last
import sympy as sp
import numpy as np
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
from scipy.integrate import odeint 
from inspect import isfunction
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import inspect

from zmq import MAX_SOCKETS
x,y,t = sp.symbols('x,y,t', real=True)

def euler_step(dxdt,x,stepsize,t):
    return dxdt(t,x)*stepsize
def rk4step(f,x,stepsize,t):
    
    h=stepsize
    k1 = h*f(t,x)
    k2 = h*f(t+h/2,x+h*k1/2)
    k3 = h*f(t+h/2,x+h*k2/2)
    k4 = h*f(t+h,x+h*k3)
    return 1/6*(k1+2*k2+2*k3+k4)
       
def wrapperforfsolveevent(t,f=None,sol=None,lastt=None,event=None):
    event(Solve_to(f,sol,lastt,t))
def Solve_ode(f,tvals,y0,method=rk4step,event=None):
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
                tguess=(tvals[i]+tvals[i-1])/2
                def wrapperforfsolveevent(t,f=f,sol=sol[i],lastt=tvals[i-1],event=event):
                    
                    return event(Solve_to(f,sol,lastt,t)[0][-1],t)
                eventT.append(fsolve(wrapperforfsolveevent,tguess))
                
                """ if t[0]-t[1] >1*10**(-15):
                    inbetweensteps = Solve_to(f,sol[i],tvals[i],tvals[i+1],method=rk4step,event=event)#replace deltattmax with maximum effective accuraccy 
                #Solves ode between steps at maximum accuracy to find more exact event locations
                for i in inbetweensteps:
                    if abs(event(i[0],i[1]))<1*10**(-14):
                        eventT.append(i[1])
                        break#if theres multiple sign change in this sign change this causes to miss them."""
        return [sol,eventT]    
    for i in range(steps-1):
        sol[i+1]=method(f,sol[i],tvals[i+1]-tvals[i],tvals[i])+sol[i]
    
    return sol


def Solve_to(f,x0,t0,tn,deltat_max=None,method=rk4step,initialvalue = True ,t_innitial_condition = None,event=None):
    if deltat_max ==None:
        deltat_max=(t0-tn)/150
    if type(f)==str:
        f= sp.lambdify([x,t],sp.parse_expr(f))
    
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
    







def findcycle(X,f,t0=0,period=None,phasecond=lambda  y : y[1]):
    if period == None:
        def G(x0,t0=t0,phasecond=phasecond):
            tn= x0[0]
            phasecond=phasecond(x0)
            x_array = solve_ivp(f,(t0,tn),x0[1:],max_step = 0.01)
            return np.append(x0[1:]-x_array.y[:,-1],phasecond)
    else:     
        def G(x0,T=period):
            t_array = [t0,T]
            x_array = solve_ivp(f,(t_array[0],t_array[1]),x0,max_step = 0.01)
            return x0[1:]-x_array.y[:,-1]
    return fsolve(G,X)

def numericalcount(f,X0,parrange,step_size=0.01, solver=fsolve):
    max_steps = int((parrange[0]-parrange[1])/step_size)
    #add somthing that masks any function with  a t parameter to same function with no t or t included in X
    #needed for consistency
    if max_steps<0:
        step_size=step_size*-1
        max_steps=max_steps*-1
    #sol = np.zeros(shape=(len(X0),max_steps)).T
    sol=np.array([X0])
    par=np.linspace(parrange[0],parrange[1],max_steps)  
    #sol[0] = 
    for i in range(max_steps-1):
        sol=np.append(sol,solver(f,sol[i],args=(par[i])))
        
        #print(solver(f,sol[i],args=(par[i])))
        
    return par,np.array(sol)



def wrapperforfsolve_no_t(X,F=None):
    return F(0,X)
def wrapperforfsolve_yes_t(X,F=None):
    t=X[0]
    X=X[1:]
    return F(t,X)
"""inputs Bellow"""




def mass_spring(t,x,const = [0.1,0.1,32,0.1,0.1]):
    m=const[0]
    gamma=const[1]
    w=const[2]
    c=const[3]
    k=const[4]
    dx1 = x[1]
    dx2 = (1/m)*(gamma*np.sin(w*t)-(c*x[1])-(k*x[0]))
    return np.array([dx1,dx2])

def drdt(X,param=0.1):
    print("in drdt")
    print(X)
    a=param
    r=X[0]
    theta =1 
    return np.array([a*r+r**3-r**5,theta])


def fsolvenotshit(f,numsolutions,domain,params=None,checks=10):
    
    #domain has 2 values for each dimension of fs input
    ranges=[]
    for i in domain:
        ranges.append(np.linspace(i[0],i[1],checks))
    #list of lists of input vals
    solutions = []
    #print(params)
    for i in ranges:
        for j in i:
            #print(j)
            sol=fsolve(f,j)
            if sol[0] not in solutions:
                solutions.append(sol[0])
            if len(solutions)==numsolutions:
                
                return [i for i in solutions if i]
    print("did not find intended number of solutions")
    return [i for i in solutions if i]
def poly(X,param=1):
    p=np.array([float(X**2+param*X+param)])
    return p
def arclengthcountinuation(f,X0,X1,parrange,step_size=0.01, solver=fsolve):
    max_steps = int((parrange[0]-parrange[1])/step_size)
    if max_steps<0:
        step_size=step_size*-1
        max_steps=max_steps*-1
    #sol = np.zeros(shape=(len(X0),max_steps)).T
    #sol[0]=X0
    #sol[1]=X1
    
    if len(X0)==1:
        X0=X0[0]
        X1=X1[0]
    sol=[X0,X1]
    print(sol)
    par=np.linspace(parrange[0],parrange[1],max_steps)
    
    def G(X2,par,L):#takes 2 previous values to guess new 
        print("\nin G")
        print(X2)
        X0=X2[0:L]
        X1=X2[L:-1]
        X2=X0
        print(X0)
        print(X1)
        secant = X1-X0
        Xprime = X1+secant
        #print(np.dot(secant,X2-Xprime))
        dot=np.dot(secant,X2-Xprime)
        feval= f(X2,par)
        g=np.array([dot,feval])
        print(g)
        return g
    for i in range(max_steps-1):
        print("here")
        
        input=[sol[i],sol[i+1]]
        print(input)
        sol=np.append(sol,fsolve(G,input,args=(par[i],len(sol[i]))))
    temp=[]
    for i in range(len(sol)):
        if i%2==0:
            temp.append(sol[i])
    sol=np.array(temp)
    return par,sol
def hopf(U,param=1):
    b=param
    print("hopf")
    print(U)
    print(param)
    du1=b*U[0]-U[1]+U[0]*(U[0]**2+U[1]**2)-U[0]*(U[0]**2+U[1]**2)**2
    du2=U[0]+b*U[1]+U[1]*(U[0]**2+U[1]**2)-U[1]*(U[0]**2+U[1]**2)**2
    return np.array([du1,du2])
if __name__ == "__main__":
    #b= findcycle([1,2,3],mass_spring)
    
    fig, axs = plt.subplots(2)
    asol=arclengthcountinuation(hopf,[1,1],[1.1,1.1],(-4,4))
    sol=numericalcount(hopf,[1,1],(-4,4))
    
    (xvals,fvals)=sol
    (axvals,afvals)=asol
    #fvals=sol
    #print(np.shape(xvals))
    #print(np.shape(fvals[0]))
    axs[0].plot(xvals,fvals)
    axs[1].plot(axvals,afvals)
    for i in axs:
        i.grid()
    #original = f(tvals,0)

    #plt.plot(tvals,original)
    plt.show()
