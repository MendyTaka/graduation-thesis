import numpy as np
import os
from matplotlib import pyplot as plt
from util import(
    Omega,
    v0,
    Rc,
    q,
    dirgraph,
    dirnum,
    Phi
)
#等高線描写
x = np.linspace(-1.5, 1.5, 1000)
y = np.linspace(-1.5, 1.5, 1000)
X, Y = np.meshgrid(x, y)



def Lagrange():
    L12 = (v0**2)/(Omega**2) - Rc**2
    L45 = (v0**2)/(Omega**2) - (q*Rc)**2
    return np.array([[np.sqrt(L12), -np.sqrt(L12), 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, np.sqrt(L45), -np.sqrt(L45)]])

if __name__=='__main__':
    x_L = Lagrange()
    plt.plot(x_L[0],x_L[1], '.k')
    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)
    Z = Phi
    plt.contourf(X, Y, Z,levels=30, cmap='gray',alpha = 0.5)
    plt.xlabel('$\it{x}$',fontsize=25,fontstyle='italic')
    plt.ylabel('y',fontsize=25)
    plt.gca().set_aspect('equal')
    plt.show()