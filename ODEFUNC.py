
from cProfile import label
from operator import add
from pickletools import ArgumentDescriptor
from turtle import st
from typing import Tuple
from xml.sax.handler import feature_validation
#from more_itertools import last
#from pytest import approx
import sympy as sp
import numpy as np
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
from scipy.integrate import odeint 
from scipy.optimize import fsolve,root
from scipy.integrate import solve_ivp
import itertools
from inspect import signature
x,y,t,p = sp.symbols('x,y,t,p', real=True)

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
def Solve_ode(f,tvals,y0,method=rk4step,event=None,termiateatevent=False):
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
                eventT.append(t)
                if termiateatevent == True:
                    print("terminating because of event")
                    return [sol,eventT]
                if type(termiateatevent) == type(1):
                    termiateatevent+=1
                    if len(eventT)>= termiateatevent:

                        print("terminating because of event")
                        return [sol,eventT]
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
    

def findroot(f,X0,methods=('hybr', 'lm'),args=None):
    if args !=None:
        for  i in range(len(methods)):
            sol = root(f, X0, method=methods[i], args=(args,))#somehow having the single argument be 2 tuples deep makes it work

            if sol.success:
                return sol.x
        print('No solution found with current method selection')
        return None
    else:
        for  i in range(len(methods)):
            sol = root(f, X0, method=methods[i])
            if sol.success:
                return sol.x
        print('No solution found with current method selection')
        return None







def numericalcount(f,X0,parrange,step_size=0.01, solver=findroot):
    max_steps = int((parrange[1]-parrange[0])/step_size)
    #add somthing that masks any function with  a t parameter to same function with no t or t included in X
    #needed for consistency
    if max_steps<0:
        step_size=step_size*-1
        max_steps=max_steps*-1
    #sol = np.zeros(shape=(len(X0),max_steps)).T
    
    sol=[X0]*max_steps
    par=np.linspace(parrange[0],parrange[1],max_steps)  
    for i in range(max_steps-1):
        print(sol[i])
        sol[i+1]=solver(f,sol[i],args=(par[i]))
        
        #print(solver(f,sol[i],args=(par[i])))
    sol=np.array(sol)
    return sol



def wrapperforfsolve_no_t(X,F):
    return F(0,X)
def wrapperforfsolve_yes_t(X,F):
    t=X[0]
    X=X[1:]
    return F(t,X)
"""inputs Bellow"""





def drdt(X,param=0.1):
    a=param
    r=X[0]
    theta =1 
    return np.array([a*r+r**3-r**5])



def poly(X,param=-1):
    #p=np.array([float(((X+param*1)*(X+param*3)*(X+param*4)))])
    p=X**3-X+param
    p=np.array([float(p)])
    return p



def arclengthcontinuation(f,X0,X1,parrange,discretisation=lambda x: x, step_size =0.1, solver=findroot,chaoticthreshold=None,max_steps=None):
    """Discretization determines what about the function f is analysed in case of default it is the roots/solutions of f
    whatever is being looked for by the discretisation function should be a solution to the discretization function
    the discretisation function is a function of f
    if trying to find cycles set, FindCycles() root finder to None and pass as discretisation function
    X0 and X1 must be correct shape for discretisation(f(X)), in case of shooting cycle finder this is 
    periodguess then guess for starting point of cycle, length: len(f)+1"""
    if max_steps == None:
        max_steps = abs(int((parrange[1]-parrange[0])/step_size))*100
    shape=np.shape(X0)
    sol=[[0]*(shape[0]+1)]*max_steps
    sol[0]=[*X0,parrange[0]]
    sol[1]=[*X1,parrange[0]+step_size]
    if parrange[1]<parrange[0]:#if backwards pass flip so logic to quit when out of par range works
        parrange=np.flip(parrange)
        print()
        print("fliping")
    print(parrange)
    print(sol[0])
    print(sol[1])
    try:#i dont like try except seems like cheating, but this is the most flexible way to do discretization
        f=discretisation(f)
    except:
        f=discretisation(f,X0)
    sol=np.array(sol)
    g=[0]*(shape[0]+1)
    for i in range(0,max_steps-2):
        secant = sol[i+1]-sol[i]
        Xprime = sol[i+1]+secant
        def G(X2,g=g,secant=secant,Xprime=Xprime):#takes 2 previous values to guess new 
            par=X2[-1]
            dot=np.dot(secant,X2-Xprime)
            feval= f(X2[:-1],(par,))
            g[0]=dot
            for i in range(len(feval)):
                g[1+i]=feval[i]
            return g
        soli=solver(G,Xprime)
        if soli is None:
            print("\n\n\nSolver cant find solution at")
            print(i)
            print(sol[i])
            return np.array(sol)
        
        sol[i+2]=soli
        
        if i % 4 == 0 and i > 32:
            print(i)
            print(soli[-1])
            if soli[-1] < parrange[0] or soli[-1]>parrange[1]:
                print(i)
                print(soli)
                print("finished parrange")
                return np.array(sol[:i])
            #if avg < chaoticthreshold*(np.sum(abs(sol[i+1]-sol[i+2]))):
            if not chaoticthreshold is None and np.sum(abs(sol[i+1]-sol[i+2]))>abs(chaoticthreshold*step_size):#using sum instead of a more common norm to avoid the square root
                print("\n\n\n\nstoping countinuation at")
                print(i)
                print(soli)
                print(np.shape(sol[i]))
                print()
                return np.array(sol[:i])
        #avg=approxRollingAverage(avg,np.sum(abs(sol[i+1]-sol[i+2])))
    print(i)
    print("completed all steps")
    return np.array(sol)

def approxRollingAverage (avg, new_sample,N=4):

    avg -= avg / N
    avg += new_sample / N

    return avg

def hopf(U,param=999):
    print(param)
    b=param
    du1=b*U[0]-U[1]+U[0]*(U[0]**2+U[1]**2)-U[0]*(U[0]**2+U[1]**2)**2
    du2=U[0]+b*U[1]+U[1]*(U[0]**2+U[1]**2)-U[1]*(U[0]**2+U[1]**2)**2

    return np.array([du1,du2])



def alreadygotsol(sol,sols,relsolsamtol=1e-06):
    if len(sols)==0:
        return False
    for i in sols:  
        if np.allclose(sol,i,rtol=relsolsamtol,atol=0.1):
            return True
    return False

def getallsolutions(f,domain,params=0,checks=100,numsolutions=None,solver=findroot,relsolsamtol=0.1):
    #domain has 2 values for each dimension of fs input
    
    ranges=[]
    for i in domain:
        ranges.append(np.linspace(i[0],i[1],checks))
    #list of lists of input vals
    combinations= itertools.product(*ranges)
    solutions = []
    
    for i in combinations:
        
        sol=solver(f,i,args=params)
       
        if all(sol==None):
            pass
       
        if ((not alreadygotsol(sol,solutions,relsolsamtol=relsolsamtol))):#any returns true if any have true
            solutions.append(sol)
            
        if len(solutions)==numsolutions:  
            print("found requested num of sols")

            return [i for i in solutions if all(i)]
    if numsolutions != None:
        print("did not find intended number of solutions")
    return [i for i in solutions if all(i)]




def addt(f,args):
    #print(args)
    def fwitht(t,x,args=args):#for inputing into functions that require t, when no t depend
            print(args)
            return f(x,args)
    return fwitht


def findcycle(func,X,t0=0,period=None,phasecond=lambda  y : y[1],args=(),rootfinder=None):#numerical shooting
    #recomended root finder "findroot" set to None for using in continuation
    if not list(signature(func).parameters)[0] in ["t","T","Time","time"]:
        f=addt(func,args)#adds dummy t dependence to function so solvivp wont complain
    else:
        f=func
    if period == None:
        def G(x0,args=args,t0=t0,phasecond=phasecond):
            tn= x0[0]
            phasecond=phasecond(x0)
            x_array = solve_ivp(f,(t0,tn),x0[1:],max_step = 0.01,args= args)
            return np.append(x0[1:]-x_array.y[:,-1],phasecond)
    else:     
        def G(x0,args=args,T=period):
            t_array = [t0,T]
            x_array = solve_ivp(f,(t_array[0],t_array[1]),x0,max_step = 0.01,args=args)
            return x0[1:]-x_array.y[:,-1]
    if rootfinder==None:
        return G
    return findroot(G,X)


def mass_spring(t,x,const = [0.1,0.1,32,0.1]):
    m=const[0]
    gamma=const[1]
    w=const[2]
    c=const[3]
    k=const[4]
    dx1 = x[1]
    dx2 = (1/m)*(gamma*np.sin(w*t)-(c*x[1])-(k*x[0]))
    return np.array([dx1,dx2])


def remove_duplicates(item_list):
    ''' Removes duplicate items from a list 
    even if list of unhashable items'''
    
    singles_list = []
    singles_liststr = []
    for element in item_list:
        if str(element) not in singles_liststr:
            singles_liststr.append(str(element))
            singles_list.append(element)
        
    return singles_list


def getbifdiagram(f,xrange,prange ,stepssize=0.01,numstartingpoints=4,plot=True,solver=findroot):
    #numstarting point is number of arc lengthcontinuation starting points per 
    sols = getallsolutions(f,xrange,params=prange[0],numsolutions=numstartingpoints,solver=solver)#find set of solutions numsolutions None = find as many as it can
    sols2=[]
    for i in sols:
        sols2.append(solver(f,i,args=(prange[0]+stepssize)))#finds second solution for each starting point

    solsb = getallsolutions(f,xrange,params=prange[1],numsolutions=numstartingpoints,solver=solver)#do the same from the other side of the domain
    #do same but starting on other end of domain
    sols2b=[]
    for i in solsb:
        sols2b.append(solver(f,i,args=(prange[1]-stepssize)))

    #sols is solutions with first parameter value
    #sols2 is solutions with second parameter value

    #solsb is solutions with last parameter value
    #sols2b is solutions with second to last parameter value
    #b indicates backwards

    print("starting values:")
    print(sols)
    print(sols2)
    print("backwards starting values:")
    print(solsb)
    print(sols2b)
    solcombinations=[]
    solcombinationsb=[]



    for i in range(len(sols)):
        solcombinations.append([sols[i],sols2[i]])#combinations of points for comming from front of domain
    for i in range(len(solsb)):
        solcombinationsb.append([solsb[i],sols2b[i]])#combinations of points used for coming from back of domain
    asol=[]
    psol=[]
    for i in solcombinations:
        A=arclengthcontinuation(f,i[0],i[1],prange,step_size=stepssize,solver=solver)#forwards pass
        asol.append(A) 
        #psol.append(A[-1])
    for j in solcombinationsb:
        A=arclengthcontinuation(f,j[0],j[1],np.flip(prange),step_size=-stepssize,solver=solver)#backwards pass
        print(np.shape(A))
        p=np.flip(A[:,-1])
        a=np.flip(A[:,:-1])
        A=[*a,p]
        print(np.shape(A))
        #psol.append(np.flip(A[-1]))
        #asol.append(np.flip(arclengthcountinuation(f,j[0],j[1],np.flip(prange),step_size=-stepssize,chaoticthreshold=chaoticthresh,solver=solver)))#back wards pass
    
    #plotting
    if plot:
        figureshape =(2,2)
        fig, axs = plt.subplots(*figureshape)
        fig.set_figheight(14)
        fig.set_figwidth(19)
        
        #fvals[-1] = parameter values 
        #asol=asol.T
        for fvals in asol:
            #axs.flatten()[0].plot(xvals,*fvals.T)
            axs.flatten()[0].plot(fvals)
            axs.flatten()[1].plot(fvals[:,-1],fvals[:,0])
            axs.flatten()[1].plot(fvals[:,-1],fvals[:,1])
            axs.flatten()[2].scatter(fvals[:,0],fvals[:,1])
            axs.flatten()[3].scatter(fvals[:,-1],fvals[:,1])
            axs.flatten()[3].scatter(fvals[:,-1],fvals[:,0])
        for i in axs.flatten():
            i.grid()
        plt.show()
    return fvals

def xdotdot(t, X):

    X_dot = np.array([X[1], -X[0]])
    return X_dot

def dx_dt2(t, X, a=0.1, b=0.1, d=0.1):
    x = X[0]
    y = X[1]

    dxdt = x*(1-x) - a*x*y/(d+x)
    dydt = b *y *(1 - y/x)

    dXdt = np.array([dxdt, dydt])
    return dXdt
def hopft(t,U,param=1):
    b=param
    du1=b*U[0]-U[1]+U[0]*(U[0]**2+U[1]**2)-U[0]*(U[0]**2+U[1]**2)**2
    du2=U[0]+b*U[1]+U[1]*(U[0]**2+U[1]**2)-U[1]*(U[0]**2+U[1]**2)**2
    return np.array([du1,du2])
import time


if __name__ == "__main__":
    
    p0=0
    step=0.1
    g1 = findcycle(hopft,(10,1,1),args=(p0,))
    g2 = findcycle(hopft,(10,1,1),args=(p0+step,))
    s1 = findroot(g1,(10,1,1),args=(p0,))
    #print(s1)
    soli=solve_ivp(hopft,(0,s1[0]),(s1[1],s1[2]),args=(p0,))
   
    plt.plot(soli.y[0],soli.y[1])
    plt.show()
    
    s2 = findroot(g2,(10,1,1),args=(p0+0.1,))
    A=arclengthcontinuation(hopft,s1,s2,(p0,1),discretisation=findcycle,step_size=step)
    print(A)
    print(np.shape(A))
    figureshape =(3,1)
    fig, axs = plt.subplots(1)
    fig.set_figheight(14)
    fig.set_figwidth(19)
    cycles=[]
    k=0
    for i in A[1::4]:
        print(i)
        soli=solve_ivp(hopft,(0,i[0]),(i[1],i[2]),args=(i[-1],))
        cycles.append(soli)
        axs.plot(soli.y[0],soli.y[1],label='%s' % float('%.4g' % i[-1]))
        k+=1
        print(k)
        
        print("drawn out of")
        print(len(A[1::8]))
    plt.legend()
    plt.show()
    
    """
    
    
    plt.show()

    
    timet=0
    timenot=0
    for i in range(100):
        tic = time.perf_counter()
        findcycle(hopf,(1,2,3),args=(i,))
        toc = time.perf_counter()
        #print(f"Found Cycle no t in {toc - tic:0.4f} seconds")
        timenot+=(toc - tic)
        tic = time.perf_counter()
        findcycle(hopft,(1,2,3),args=(i,))
        toc = time.perf_counter()
        timet+=(toc - tic)
        #print(f"Found Cycle with t in {toc - tic:0.4f} seconds")
    print(timet/100)
    print(timenot/100)"""

    """ polysol1 = findroot(poly,10,args=-2)
    polysol2 = findroot(poly,10,args=-2+0.01)
   
    A= arclengthcontinuation(poly,polysol2,polysol1,(-2,2),step_size=0.01)
    print("first 30 a")
    
    figureshape =(2,1)
    fig, axs = plt.subplots(*figureshape)
    F=x**3-x+p
    s= sp.latex(F)
    axs[0].text(0,1.01,"solutions to: "+('$%s$'%s)+"  for p in [-2,2]",transform=axs[0].transAxes)
    axs[0].plot(A[:,1],A[:,0])
    axs[1].plot(A)
    
    plt.show()"""
    """ p=[0.1,0.01,0.001,0.0001]
    plt.rcParams.update({'font.size': 22})
    #print(np.shape(y))
    figureshape =(2,2)
    
    
   
    fig, axs = plt.subplots(*figureshape)
    fig.set_figheight(14)
    fig.set_figwidth(19)
    fig.suptitle("t = 0 and 40 innitial value = (1,1)")
    p=[0.1,0.01,0.001,0.0001]
    
    for i in range(len(p)):
        print(p[i])
        y=Solve_to(dx_dt2,[1,1],0,40,p[i])
        (xvals,tvals) = y
        axs.flatten()[i].plot(xvals[:,0],xvals[:,1])
        axs.flatten()[i].title.set_text("Time step:" + str(p[i]))
    plt.show()"""
    """
    print(x0)
    print(x1)
    A=arclengthcountinuation(poly,x0,x1,parrange=(-4,4),step_size=0.01)
    parrange=(0,15)
    step_size=0.01
    max_steps = int((parrange[1]-parrange[0])/step_size)
    
   
    par=np.linspace(parrange[0],parrange[1],max_steps)
    print("first 30 a")
    print(A[:30])
    
    print(np.shape(A))
    figureshape =(3,1)
    fig, axs = plt.subplots(*figureshape)
    axs[0].plot(A[:,0],A[:,1])
    axs[1].plot(A)
    A=A.T
    axs[2].plot(A[:,0],A[:,1])
    plt.show()
    print(poly(0,param=0))"""
    
