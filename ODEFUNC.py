
from operator import add
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
    









def numericalcount(f,X0,parrange,step_size=0.01, solver=fsolve):
    max_steps = int((parrange[1]-parrange[0])/step_size)
    #add somthing that masks any function with  a t parameter to same function with no t or t included in X
    #needed for consistency
    if max_steps<0:
        step_size=step_size*-1
        max_steps=max_steps*-1
    #sol = np.zeros(shape=(len(X0),max_steps)).T
    
    sol=[X0]*max_steps
    par=np.linspace(parrange[0],parrange[1],max_steps)  
    #sol[0] = 
    
    for i in range(max_steps-1):
        #print(sol[i])
        sol[i+1]=solver(f,sol[i],args=(par[i]))
        
        #print(solver(f,sol[i],args=(par[i])))
    sol=np.array(sol)
    return np.array([par[:],sol[:,0]])



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
    p=np.array([float(((X+param*1)*(X+param*3)*(X+param*4)))])
    #p=np.array([float((-param*X)**3+X**2+-4*X+param)])
    return p



def arclengthcountinuation(f,X0,X1,parrange,step_size=0.01, solver=fsolve,chaoticthreshold=100):
    
    max_steps = int((parrange[1]-parrange[0])/step_size)
    shape=np.shape(X0)
    """if max_steps<0:
        print("\n\n\n\n max steps less than 0 \n\n\n\n")
        step_size=step_size*-1
        max_steps=max_steps*-1"""
    if len(X0)==1:
        print("\n\n1 d problem\n\n")
        shape=(1,)
    sol=[[0]*shape[0]]*max_steps
    sol[0]=X0
    sol[1]=X1
    
    par=np.linspace(parrange[0],parrange[1],max_steps)
    g=[]
    while len(g)<=(shape[0]*2)-1:#fixes dim of output to dim of input
            g.append(0)
    def G(X2,par,g=g):#takes 2 previous values to guess new 
        L=int(par[-1])
        par=par[0]
        X0=X2[0:L]
        X1=X2[L:]
        X2=X0
        secant = X1-X0
        Xprime = X1+secant
        dot=np.dot(secant,X2-Xprime)
        feval= f(X2,par)
        #g=[dot,*feval]
        g[0]=dot
        for i in range(len(feval)):
            g[1+i]=feval[i]
        g=np.array(g).flatten()
        return g
    avg=chaoticthreshold**2
    
    for i in range(0,max_steps-2):
        #soli,infodict,ier,mesg=solver(G,[sol[i],sol[i+1]],args=[par[i],shape[0]],factor=0.1,xtol=1e-10,full_output=True)
        soli=solver(G,[sol[i+1],sol[i]],args=[par[i],shape[0]])
       
        
        if soli is None:
            print("\n\n\nSolver cant find solution at")
            print(i)
            print(sol[i])
            print(par[i])
            return np.array(sol)
        soli=soli[:shape[0]]
        sol[i+2]=soli
        assert np.shape(soli)==shape
        
        """if i % 8 == 0 and i > 8:
            #if avg < chaoticthreshold*(np.sum(abs(sol[i+1]-sol[i+2]))):
            if np.sum(abs(sol[i+1]-sol[i+2]))>abs(chaoticthreshold*step_size):#using sum instead of a more common norm to avoid the square root
                print("\n\n\n\nstoping countinuation at")
                print(i)
                print(soli)
                print(np.shape(sol[i]))
                print()
                return np.array(sol)#[0:i]"""
        #avg=approxRollingAverage(avg,np.sum(abs(sol[i+1]-sol[i+2])))
    print(np.shape(sol))   
    return np.array(sol)

def approxRollingAverage (avg, new_sample,N=4):

    avg -= avg / N
    avg += new_sample / N

    return avg

def hopf(U,param=1):
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
def findroot(f,X0,methods=('hybr', 'lm'),args=None):
    if args !=None:
        print(type(args))
        for  i in range(len(methods)):
            sol = root(f, X0, method=methods[i], args=args)
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
def getallsolutions(f,domain,params=0,checks=10,numsolutions=None,solver=findroot,relsolsamtol=0.1):
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
    #print("inaddt")
    def fwitht(t,x,args=args):#for inputing into functions that require t, when no t depend
            return f(x,args)
    return fwitht


def findcycle(func,X,t0=0,period=None,phasecond=lambda  y : y[1],args=()):
    print("infindcycle")
    
    
    if not list(signature(func).parameters)[0] in ["t","T","Time","time"]:
        f=addt(func,args)#adds dummy t dependence to function so solvivp wont complain
    else:
        f=func
    
    if period == None:
        def G(x0,t0=t0,phasecond=phasecond):
            tn= x0[0]
            phasecond=phasecond(x0)
            x_array = solve_ivp(f,(t0,tn),x0[1:],max_step = 0.01,args= args)
            return np.append(x0[1:]-x_array.y[:,-1],phasecond)
    else:     
        def G(x0,T=period):
            t_array = [t0,T]
            x_array = solve_ivp(f,(t_array[0],t_array[1]),x0,max_step = 0.01,args=args)
            return x0[1:]-x_array.y[:,-1]
    
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


def getbifdiagram(f,xrange,prange ,stepssize=0.0001,numstartingpoints=3,plot=True,chaoticthresh = 1200,solver=findroot):
    sols = getallsolutions(f,xrange,params=prange[0],numsolutions=numstartingpoints,solver=solver)#find set of solutions numsolutions None = find as many as it can
    sols2=[]
    for i in sols:
        sols2.append(fsolve(f,i,args=(prange[0]+stepssize)))#finds second solution for each starting point

    solsb = getallsolutions(f,xrange,params=prange[1],numsolutions=numstartingpoints,solver=solver)#do the same from the other side of the domain
    #do same but starting on other end of domain
    sols2b=[]
    for i in solsb:
        sols2b.append(fsolve(f,i,args=(prange[1]-stepssize)))

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

    for i in solcombinations:
        asol.append(arclengthcountinuation(f,i[0],i[1],prange,step_size=stepssize,chaoticthreshold=chaoticthresh,solver=solver)) #forwards pass
    for j in solcombinationsb:
        asol.append(np.flip(arclengthcountinuation(f,j[0],j[1],np.flip(prange),step_size=-stepssize,chaoticthreshold=chaoticthresh,solver=solver)))#back wards pass
    
    #plotting
    if plot:
        figureshape =(2,2)
        fig, axs = plt.subplots(*figureshape)
        fig.set_figheight(14)
        fig.set_figwidth(19)
        max_steps = abs(int((prange[1]-prange[0])/stepssize))
        xvals = np.linspace(*prange, max_steps)
        for fvals in asol:
            axs.flatten()[0].plot(xvals,fvals)
            axs.flatten()[1].plot(fvals[:,0],fvals[:,1])
            axs.flatten()[2].scatter(xvals,fvals[:,0],fvals[:,1])
            axs.flatten()[3].scatter(fvals[:,0],fvals[:,1])
        for i in axs.flatten():
            i.grid()
        plt.show()
    return xvals,fvals




if __name__ == "__main__":
    hopft= addt(hopf,(99,))
    c=findcycle(hopf,(5,0,1),args=(99,))
    
    print("solveivp")
    print(hopf)
    
    sol=solve_ivp(hopft,(0,c[0]),c[1:],args=(99,))
    
    figureshape =(1,3)
    #fig, axs = plt.subplots(*figureshape)
    fig=plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(2,2,1)
    ax2 = plt.subplot(2,2,2)
    ax3 = plt.subplot(2,1,2)
    axs=[ax1,ax2,ax3]
    axs[2].plot(*sol.y)
    axs[1].plot(sol.t,sol.y[0])
    axs[0].plot(sol.t,sol.y[1])
    axs[2].title.set_text("State variables")
    axs[1].title.set_text("State variable 1 against T")
    axs[0].title.set_text("State variable 2 against T")
    for i in axs:
            i.grid()
    fig.suptitle("Hopf bifurcation Limit cycle at b=99")
    plt.show()
    #print(sol)
    #x0=fsolve(hopf,(1,0),args=0.1)
    #x1=fsolve(hopf,(1,0),args=0.2)
    #print((x1,x0))
    #A=arclengthcountinuation(hopf,x0,x1,parrange=(0.1,5),step_size=0.1,solver=findcycle)
    #print(np.shape(A))
    #plt.plot(A)
    #plt.show()
    
