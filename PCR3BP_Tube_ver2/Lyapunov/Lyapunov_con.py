import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import os,time
import Lagrange, Lyapunov_init, Lyapunov_diff

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

#エネルギー計算
def Energy(mu,x):
    r1 = np.sqrt((x[0]+mu)**2+x[1]**2)
    r2 = np.sqrt((x[0]-1+mu)**2+x[1]**2)
    #エネルギー
    E = (x[2]**2+x[3]**2)/2-(x[0]**2+x[1]**2)/2 -(1-mu)/r1 -mu/r2 -mu*(1-mu)/2
    return E

#Contination
def Contination(mu,x1,x2,energy):
    energy_init = Energy(mu, x1)
    energy_re = Energy(mu, x2)
    delta = x2 - x1
    print(energy_re)
    if abs(energy_re - energy)<1e-10:
        return x2
    elif (energy_init - energy)*(energy_re - energy)<0:
        delta = delta*0.5
        x2 = Lyapunov_diff.diff_corr(mu, x1+delta)
        return Contination(mu, x1, x2, energy)
    else:
        x1 = x2
        x2 = Lyapunov_diff.diff_corr(mu, x1+delta)
        return Contination(mu, x1, x2, energy)

#L1,数値の保存も含めたcontination
def Con_init(mu, xe, energy):
    #ラグランジュ点の判別
    if xe>1-mu:
        file_name = 'L2_init.npy'
    elif xe>0:
        file_name = 'L1_init.npy'
    else:
        file_name = 'L3_init.npy'
    #値の検索
    if file_name in os.listdir(dirnum):
        initdata = np.load(dirnum+file_name, allow_pickle=True).item()
        if energy in initdata.keys():
            return initdata[energy]
    else:
        initdata = dict()

    init_x1 = Lyapunov_init.lyap_init(mu, 0.001, xe)
    init_x2 = Lyapunov_init.lyap_init(mu, 0.005, xe)
    x_diff1 = Lyapunov_diff.diff_corr(mu, init_x1)
    x_diff2 = Lyapunov_diff.diff_corr(mu, init_x2)
    x_contination = Contination(mu, x_diff1, x_diff2, energy)
    #値の保存
    initdata[energy] = x_contination
    np.save(dirnum+file_name,initdata)

    return x_contination

def main():
    t1 = time.time()
    L = Lagrange.Lagrangepoint(mu)
    E = [-1.59, -1.57, -1.55, -1.53, -1.51]
    color = ['b', 'c', 'g', 'k', 'm']

    #図の設定
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot()
    for i in range(len(E)):
        #初期条件の計算
        L1_con = Con_init(mu, L[0][0], E[i])
        L2_con = Con_init(mu, L[0][1], E[i])
        #積分計算
        y1 = Lyapunov(L1_con)
        y2 = Lyapunov(L2_con)
        ax.plot(y1[0], y1[1], color=color[i], label='E='+str(E[i]))
        ax.plot(y2[0], y2[1], color=color[i])
    ax.plot(L[0][0],L[1][0],'rx')
    ax.plot(L[0][1],L[1][1],'rx')
    ax.plot(1-mu, 0, marker='.', markersize=10, color='y')
    ax.set_xlabel('$\it{x}$',fontsize=25,fontstyle='italic')
    ax.set_ylabel('y',fontsize=25)
    fig.gca().set_aspect('equal')
    ax.set_title('Contination')
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()

    #グラフの保存
    fig.savefig(dirgraph+"Contination.png")

    t2 = time.time()
    print(f'経過時間:{t2-t1}s')
    plt.show()

if __name__=='__main__':
    main()