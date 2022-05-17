import os
import numpy as np
import matplotlib.pyplot as plt

#共通変数
from util import (
    mu,
    dirnum,
    dirgraph
)

#方程式の定義
f1 = lambda x:x**5-(3-mu)*x**4+(3-2*mu)*x**3-mu*x**2+2*mu*x-mu
f2 = lambda x:x**5+(3-mu)*x**4+(3-2*mu)*x**3-mu*x**2-2*mu*x-mu
f3 = lambda x:x**5+(2+mu)*x**4+(1+2*mu)*x**3-(1-mu)*x**2-2*(1-mu)*x-1+mu

#微分した方程式
df1 = lambda x:5*x**4 -4*(3-mu)*x**3 +3*(3-2*mu)*x**2 -2*mu*x+2*mu
df2 = lambda x:5*x**4 +4*(3-mu)*x**3 +3*(3-2*mu)*x**2 -2*mu*x-2*mu
df3 = lambda x:5*x**4 +4*(2+mu)*x**3 +3*(1+2*mu)*x**2 -2*(1-mu)*x-2*(1-mu)

#二分法
def bisec(f, a, b):
    while abs(a-b)>1e-13:
        c = (a+b)/2
        if f(a)*f(c)<0:
            b = c
        else:
            a = c
    
    return a

#ニュートン法
def Newton(f, df, a):
    while True:
        b = a - f(a)/df(a)
        if abs(b - a)<1e-13:
            break
        a = b
    return a

#ラグランジュ点を計算する(二分法)
def Lagrange_bisec(mu):
    L1 = bisec(f1,0,1)
    L2 = bisec(f2,0,1)
    L3 = bisec(f3,0,1)
            
    xpoints=np.array([1-mu-L1, 1-mu+L2, -mu-L3, 0.5-mu, 0.5-mu])
    ypoints=np.array([0, 0, 0, np.sqrt(3)/2, -np.sqrt(3)/2])
    return np.array([xpoints,ypoints])

#ラグランジュ点を計算する(ニュートン法)
def Lagrange_newton(mu):
    L1 = Newton(f1, df1, 1)
    L2 = Newton(f2, df2, 1)
    L3 = Newton(f3, df3, 1)
    #Lagrange点
    xpoints=np.array([1-mu-L1, 1-mu+L2, -mu-L3, 0.5-mu, 0.5-mu])
    ypoints=np.array([0, 0, 0, np.sqrt(3)/2, -np.sqrt(3)/2])
    
    return np.array([xpoints,ypoints])


#ライブラリーを使う方法
def Lagrange_lib(mu):
    #L1のx座標を求める
    L1_x=np.roots([1, -3+mu, 3-2*mu, -mu, 2*mu, -mu])
    #虚数解を取り除く
    for i in L1_x:
        if i.imag == 0:
            L1 = float(1-mu-i.real)
            break
    
    #L2のx座標を求める
    L2_x=np.roots([1, 3-mu, 3-2*mu, -mu, -2*mu, -mu])
    for i in L2_x:
        if i.imag == 0:
            L2 = float(1-mu+i.real)
            break
    
    #L3のx座標を求める
    L3_x=np.roots([1, 2+mu, 1+2*mu, -(1-mu), -2*(1-mu), -1+mu])
    for i in L3_x:
        if i.imag == 0:
            L3 = float(-mu-i.real)
            break
    
    #Lagrange点
    xpoints=np.array([L1, L2, L3, 0.5-mu, 0.5-mu])
    ypoints=np.array([0, 0, 0, np.sqrt(3)/2, -np.sqrt(3)/2])
    
    return np.array([xpoints,ypoints])

#Lagrange点
def Lagrangepoint(mu):
    #値を保存
    if 'Lagrange.npy' in os.listdir(dirnum):
        Lagrange = np.load(dirnum+'Lagrange.npy')
        return Lagrange
    x = Lagrange_lib(mu)
    np.save(dirnum+'Lagrange.npy', x)
    return x

def main():
    x = Lagrangepoint(mu)
    print(*x)
    #図示
    fig  = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    #P1(地球)
    ax.plot(-mu, 0, marker='.', markersize=20, color='b')
    #P2(月)
    ax.plot(1-mu, 0, marker='.', markersize=10, color='y')
    #ラグランジュ点
    ax.plot(x[0],x[1],'o', markersize=5, color='k')

    #ラベル
    ax.text(-0.18, -0.2, 'Earth',fontsize=15)
    ax.text(0.8, 0.1, 'Moon',fontsize=15)
    ax.text(x[0][0]-0.05, -0.2, '$L_1$',fontsize=15)
    ax.text(x[0][1]-0.05, -0.2, '$L_2$',fontsize=15)
    ax.text(x[0][2]-0.05, -0.2, '$L_3$',fontsize=15)
    ax.text(x[0][3]+0.05, np.sqrt(3)/2-0.1, '$L_4$',fontsize=15)
    ax.text(x[0][4]+0.05, -np.sqrt(3)/2, '$L_5$',fontsize=15)
    
    ax.set_xlabel('$\it{x}$',fontsize=25,fontstyle='italic')
    ax.set_ylabel('y',fontsize=25)
    fig.gca().set_aspect('equal')
    ax.set_title('Lagrangepoint')
    plt.tight_layout()
    #グラフの保存
    fig.savefig(dirgraph+"Lagrangepoint.png")

    plt.show()


if __name__=="__main__":
    main()