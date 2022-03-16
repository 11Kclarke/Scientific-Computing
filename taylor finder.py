# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 16:31:34 2022

@author: kiera
"""
import matplotlib as plt
import sympy as sp
import math
x,y,t = sp.symbols('x,y,t', real=True)
a,b,c,d = sp.symbols('a,b,c,d', real=True)

def taylorconverter(centre, funtion, terms):
     taylor = funtion.subs(x,centre)
     function =funtion.subs(x,centre)
     for i in range(terms):
         #print(taylor)
         Derivative= sp.diff(funtion,x,i+1).subs(x,centre)
         #print("derivative is  "+str(Derivative))
         Polynomial = (((x-centre)**(i+1))/math.factorial((i+1)))
         #print("Polynomialis  "+str(Polynomial))
         iterm = Derivative*Polynomial
         #print(str(i)+"term is  "+str(iterm))
         taylor+=iterm
     taylor = sp.simplify(taylor)   
         #sp.Poly(taylor).add(sp.Poly((sp.diff(funtion,x)).subs(x,centre)*(((x-centre)**i)/math.factorial(i)))) #impliments general taylor series formual for i number of terms
     print(sp.latex(function))
     print("=")
     print(sp.latex(taylor)+"\dots")
     sp.pprint(function)
     sp.pprint(taylor)
     return taylor

func = 1/(x-b)

import matplotlib.pyplot as plt
a = sp.latex(taylorconverter(a,func,5))
f, ax = plt.subplots(1,1,figsize=(15,15))
#ax.figsize(15,15)
ax.text(0,0.5,r"$%s$" %(a),fontsize=30,color="green")
plt.show()