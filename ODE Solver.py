# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 14:06:02 2022

@author: kiera
"""
import sympy as sp
import numpy as np
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
from scipy.integrate import odeint 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1)
x,y,t = sp.symbols('x,y,t', real=True)
f1t = sp.sin(t+t**2)
f1 = sp.sin(x+x**2)
sp.pprint(f1)

lam_x = lambdify([x,t], f1, modules=['numpy'])
lam_t = lambdify([t,x], f1, modules=['numpy'])
import inspect
print(inspect.getsource(lam_x))
steps = 802
y0=0
xrange=(-4,4)
x0, xn = xrange
xvals= np.linspace(*xrange,steps)
stepsize=abs((x0-xn)/steps) 
print(stepsize)



def euler_step(f,x,stepsize,t=0):
    return f(x,t)*stepsize


def euler_solve(f,y0,xvals,stepsize):
    steps = abs(int((xvals[0]-xvals[-1])/stepsize))
    finalstepsize = xvals[-1]-(xvals[0]+stepsize*steps)
    sol=np.zeros(steps+1) 
    sol[0]=y0
    for i in range(steps-1):
        sol[i+1]=sol[i]+euler_step(f,xvals[i],stepsize)
    sol=np.append(sol,(sol[i]+euler_step(f,xvals[-1],finalstepsize))) 
    return sol
sol = euler_solve(lam_x,y0,xvals,0.01)


ax1.plot(xvals,lam_x(xvals,0))
ax2.plot(xvals,sol)

yout, info = odeint(lam_t, y0, xvals,full_output=True)
     


def rk4step(f,x,stepsize,t=0):
    h=stepsize
    k1 = k2 = k3 = k4 = 0
    k1 = h*f(x,t)
    k2 = h*f(x+h*k1/2,t+h/2)
    k3 = h*f(x+h*k2/2,t+h/2)
    k4 = h*f(x+h*k3,t+h)
    return 1/6*(k1+2*k2+2*k3+k4)
       
def rk4(f,xvals,n,y0):
    
    h = abs(xvals[0]-xvals[1])#stepsize
    sol = np.zeros(n)
    sol[0]=y0
    for i in range(n-1):        
        sol[i+1]=rk4step(f,xvals[i],h)+sol[i]
    return sol
rk4sol= rk4(lam_x,xvals,steps,y0)    
ax3.plot(xvals,rk4sol)


#print(yout)
#print(yout)
ax4.plot(xvals,yout)
ax1.annotate("Original Function:",(0,1))
ax2.annotate("Euler method solution step size:"+str(stepsize),(0,1))
ax3.annotate("RK4 solution:",(0,1))
ax4.annotate("Sci Py solution:",(0,1)) 
plt.show()



