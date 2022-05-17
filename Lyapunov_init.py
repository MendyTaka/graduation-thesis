import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp 
import os
import Lagrange
import time

#共通変数,関数
from util import (
    mu,
    dirnum,
    dirgraph,
    func,
    event_y0pm
)

#初期条件の計算
def lyap_init(mu, Ax, xe):
    mub = mu/(abs(xe - 1 + mu)**3) + (1 - mu)/(abs(xe + mu)**3)
    nu = np.sqrt(-0.5*(mub - 2 - np.sqrt(9*mub**2 - 8*mub)))
    tau = -(nu**2 + 2*mub + 1)/(2*nu)
    x0 = xe - Ax
    vy0 = -Ax*nu*tau
    init_x = np.append(np.array([x0,0,0,vy0]),np.eye(4).reshape(1,16))
    return init_x

def main():
    t1 = time.time()
    #ラグランジュ点の座標
    L = Lagrange.Lagrangepoint(mu)
    #軌道計算
    init_x = lyap_init(mu, 0.001, L[0][0])
    line = solve_ivp(func, [0,100], init_x, events=event_y0pm,rtol = 1e-12, atol = 1e-12)
    y = line.y
    #グラフの設定
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(y[0],y[1])
    ax.plot(L[0][0],L[1][0],'kx')
    ax.set_xlabel('$\it{x}$',fontsize=25,fontstyle='italic')
    ax.set_ylabel('y',fontsize=25)
    fig.gca().set_aspect('equal')
    ax.set_title('L1_orbit_init')
    plt.tight_layout()
    #グラフの保存
    fig.savefig(dirgraph+"L1_orbit_init.png")

    t2 = time.time()
    print(f'経過時間:{t2-t1}s')
    plt.show()

if __name__=='__main__':
    main()