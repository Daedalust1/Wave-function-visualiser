#!/usr/bin/env python
# coding: utf-8

# In[7]:


from scipy.sparse import *
from scipy.sparse.linalg import *
from mpl_toolkits.mplot3d import *
import numpy as np
import matplotlib.pyplot as plt
def wave_function_plotter(N=100,num_states=5):
    #intialising the grid
    x=np.linspace(0,1e-9,N)
    X,Y=np.meshgrid(x,x)
    delta=x[1]-x[0]
    #defining the constants
    h_bar=1.0545*(10**(-34))
    me=9.1*(10**(-31))
    #definiing the laplacian numerically 
    mdiag=-2*np.ones(N)
    odiag=np.ones(N-1)
    lap1d=diags([odiag,mdiag,odiag],[-1,0,1])/delta**2
    I=eye(N)
    lap2d=kron(lap1d,I)+kron(I,lap1d)
    #defining the potential energy (2D infinte square well) and the Hamiltionian
    V=np.zeros([N,N])
    Vf=V.flatten()
    H=(-h_bar**2/(2*me))*lap2d + diags(Vf)
    #solving the Time Independent Schrodinger Equation using eigen values
    E,psi=eigsh(H,k=num_states+1,which='SM')
    eig=np.argsort(E.real)
    E=E[eig]
    psi=psi[:,eig]
    #converting the energy into electon volts
    ev=1.6e-19
    for i in range(num_states+1):
        #intialising the 3D graph
        fig=plt.figure()
        ax=fig.add_subplot(111, projection='3d')
        #plotting probability, |psi|^2
        psi_i=psi[:, i].real
        psi_i=(psi_i/np.sqrt(np.sum(np.abs(psi_i)**2)*delta**2)).reshape(N,N)
        P=psi_i**2
        ax.plot_surface(X,Y,P,cmap='twilight_shifted')
        #making the graph more understandable 
        ax.view_init(elev=75,azim=45)
        plt.title(f"State {i} of probability density")
        plt.xlabel("x (m)")
        plt.ylabel("|ψ(x,y)|^2")
        plt.show()
        #calculating the allowed energies for each state
        print(f"The energies in terms of electron volt are{E.real/ev}")


# In[9]:


wave_function_plotter()


# In[ ]:




