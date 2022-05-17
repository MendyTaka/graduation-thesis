import numpy as np
import os
from matplotlib import pyplot as plt

#パラメータの設定
mu = 0.01215

#等高線描写
x = np.linspace(-2, 2, 1000)
y = np.linspace(-2, 2, 1000)
X, Y = np.meshgrid(x, y)
r1 = np.sqrt((X+mu)**2+Y**2)
r2 = np.sqrt((X-1+mu)**2+Y**2)
U = -1/2*(X**2+Y**2)-(1-mu)/r1-mu/r2-mu*(1-mu)/2

#ラグランジュ点を計算する
#二分法
def Lagrange_bisec(mu):
    #方程式の定義
    f1 = lambda x:x**5-(3-mu)*x**4+(3-2*mu)*x**3-mu*x**2+2*mu*x-mu
    f2 = lambda x:x**5+(3-mu)*x**4+(3-2*mu)*x**3-mu*x**2-2*mu*x-mu
    f3 = lambda x:x**5+(2+mu)*x**4+(1+2*mu)*x**3-(1-mu)*x**2-2*(1-mu)*x-1+mu
    #初期値の定義
    L1_a = 0
    L1_b = 1
    L2_a = 0
    L2_b = 1
    L3_a = 0
    L3_b = 1
    #二分法
    #L1
    while abs(L1_a-L1_b)>1e-8:
        L1_c = (L1_a+L1_b)/2
        if f1(L1_a)*f1(L1_c)<0:
            L1_b = L1_c
        else:
            L1_a = L1_c
    #L2
    while abs(L2_a-L2_b)>1e-8:
        L2_c = (L2_a+L2_b)/2
        if f2(L2_a)*f2(L2_c)<0:
            L2_b = L2_c
        else:
            L2_a = L2_c
    #L3
    while abs(L3_a-L3_b)>1e-8:
        L3_c = (L3_a+L3_b)/2
        if f3(L3_a)*f3(L3_c)<0:
            L3_b = L3_c
        else:
            L3_a = L3_c
            
    xpoints=np.array([1-mu-L1_a, 1-mu+L2_a, -mu-L3_a, 0.5-mu, 0.5-mu])
    ypoints=np.array([0, 0, 0, np.sqrt(3)/2, -np.sqrt(3)/2])
    
    return np.array([xpoints,ypoints])

if __name__=='__main__':
    x_L = Lagrange_bisec(mu)
    plt.plot(x_L[0],x_L[1], '.k')
    plt.xlim(-2.0,2.0)
    plt.ylim(-2.0,2.0)
    Z = U
    plt.contourf(X, Y, Z, alpha = 0.5)
    plt.xlabel('$\it{x}$',fontsize=25,fontstyle='italic')
    plt.ylabel('y',fontsize=25)
    plt.gca().set_aspect('equal')
    plt.show()