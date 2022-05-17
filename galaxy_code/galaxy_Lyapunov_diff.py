import numpy as np
import matplotlib.pyplot as plt
import time, os

from util import(
    Omega,
    v0,
    Rc,
    q,
    dirgraph,
    dirnum,
    Energy,
    Phixx,
    Phiyy,
    func,
    Lagrange,
    Lyapunov_half,
    Lyapunov
)
#初期値計算
def lyap_init(xe, Ax=1e-5):
    alpha = 4*Omega**2 + Phixx(xe, 0) + Phiyy(xe, 0)
    beta = (-alpha-np.sqrt(alpha**2 -4*Phixx(xe, 0)*Phiyy(xe, 0)))/2
    nu = np.sqrt(-beta)
    tau = (Phixx(xe, 0)-nu**2)/(2*Omega*nu)
    x0 = xe - Ax
    vy0 = -Ax*nu*tau
    init_x = np.append(np.array([x0,0,0,vy0]),np.eye(4).reshape(1,16))
    return init_x

#微分補正
def diff_corr(x):
    #数値積分
    x_fin = Lyapunov_half(x)
    if abs(x_fin[2][-1]) <=1e-12:
        return x
    else:
        vx_fin = func(1,np.array([x_fin[i][-1] for i in range(len(x_fin))]))
        dvy0 = x_fin[2][-1]/(x_fin[15][-1] -vx_fin[2]*x_fin[11][-1]/x_fin[3][-1])
        x_new = np.array([0,0,0,-dvy0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        return diff_corr(x+x_new)

def main():
    t1 = time.time()
    #ラグランジュ点の座標
    L = Lagrange()
    #初期条件
    init_x = lyap_init(L[0][0])
    diff_x = diff_corr(init_x)
    #数値計算
    y0 = Lyapunov(init_x)
    y1 = Lyapunov(diff_x)
    
    #グラフの設定
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(y0[0],y0[1],label='init')
    ax.plot(y1[0],y1[1],label='diff_corr')
    ax.plot(L[0][0],L[1][0],'k.')
    ax.set_xlabel('$\it{x}$',fontsize=25,fontstyle='italic')
    ax.set_ylabel('y',fontsize=25)
    fig.gca().set_aspect('equal')
    plt.tight_layout()
    plt.legend(loc='upper right')
    #グラフの保存
    fig.savefig(dirgraph+"Lyapunov.png")

    t2 = time.time()
    print(f'経過時間:{t2-t1}s')
    plt.show()

def main2():
    t1 = time.time()
    #ラグランジュ点の座標
    L = Lagrange()
    N=100
    init_E = list()
    diff_E = list()
    #初期のずれ
    Ax = np.linspace(1e-4,1,N)
    for i in range(N):
        #初期条件
        init_x = lyap_init(L[0][0],Ax[i])
        #微分補正
        x_diff = diff_corr(init_x)
        init_E.append(Energy(init_x))
        diff_E.append(Energy(x_diff))

    #グラフの設定
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(Ax,init_E,label='init')
    ax.plot(Ax,diff_E,label='diff_corr')
    ax.set_xlabel('$\it{Ax}$',fontsize=25,fontstyle='italic')
    ax.set_ylabel('Energy',fontsize=25)
    #fig.gca().set_aspect('equal')
    ax.set_title('Ax_energy')
    plt.tight_layout()
    plt.legend()
    #グラフの保存
    fig.savefig(dirgraph+'Ax_energy.png')
    

    t2 = time.time()
    print(f'経過時間:{t2-t1}s')
    
    plt.show()

if __name__=='__main__':
    main()
    main2()