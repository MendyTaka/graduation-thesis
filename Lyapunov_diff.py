import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp 
import os,time
import Lagrange, Lyapunov_init

#共通変数,関数
from util import (
    mu,
    dirnum,
    dirgraph,
    func,
    event_y0pm,
    event_y0mp,
    Lyapunov_half,
    Lyapunov
)


#微分補正
def diff_corr(mu,x):
    #数値積分
    x_fin = Lyapunov_half(x)
    if abs(x_fin[2][-1]) <=1e-12:
        return x
    else:
        r1 = np.sqrt((x_fin[0][-1]+mu)**2+x_fin[1][-1]**2)
        r2 = np.sqrt((x_fin[0][-1]-1+mu)**2+x_fin[1][-1]**2)
        Ux = -x_fin[0][-1]+(1 - mu)*(x_fin[0][-1] + mu)/r1**3+mu*(x_fin[0][-1]-1 +mu)/r2**3
        dvy0 = ((x_fin[15][-1] -(2*x_fin[3][-1] - Ux)*x_fin[11][-1]/x_fin[3][-1])**(-1))*x_fin[2][-1]
        x_new = np.array([0,0,0,-dvy0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        return diff_corr(mu, x+x_new)

def main():
    t1 = time.time()
    L = Lagrange.Lagrangepoint(mu)
    #L1周りの初期条件
    init_x = Lyapunov_init.lyap_init(mu, 0.001, L[0][0])
    x_diff = diff_corr(mu,init_x)

    #軌道計算
    y_init = Lyapunov(init_x)
    y_diff = Lyapunov(x_diff)

    #図示
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    ax.plot(y_init[0],y_init[1], label='init')
    ax.plot(y_diff[0],y_diff[1], label='diffcorr')
    ax.plot(L[0][0],L[1][0],'kX')
    ax.set_xlabel('$\it{x}$',fontsize=25,fontstyle='italic')
    ax.set_ylabel('y',fontsize=25)
    fig.gca().set_aspect('equal')
    ax.set_title('L1_diffcorr')
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()

    #グラフの保存
    fig.savefig(dirgraph+"L1_diffcorr.png")
    t2 = time.time()
    print(f'経過時間:{t2-t1}s')
    plt.show()


if __name__=='__main__':
    main()