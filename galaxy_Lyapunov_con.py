import numpy as np
import matplotlib.pyplot as plt
import time, os
from Lyapunov_diff import(
    diff_corr,
    lyap_init
)

from util import(
    Omega,
    v0,
    Rc,
    q,
    dirgraph,
    dirnum,
    Energy,
    Phi,
    func,
    Lagrange,
    Lyapunov_half,
    Lyapunov
)

#Contination
def Contination(x1,x2,energy):
    energy_init = Energy(x1)
    energy_re = Energy(x2)
    delta = x2 - x1
    print(energy_re)
    if abs(energy_re - energy)<1e-12:
        return x2
    elif (energy_init - energy)*(energy_re - energy)<0:
        delta = delta*0.5
        x2 = diff_corr(x1+delta)
        return Contination(x1, x2, energy)
    else:
        x1 = x2
        x2 = diff_corr(x1+delta)
        return Contination(x1, x2, energy)

#L1,数値の保存も含めたcontination
def Con_init(xe, energy):
    #ラグランジュ点の判別
    file_name = 'L1_init.npy' if xe>0 else 'L2_init.npy'
    #値の検索
    if file_name in os.listdir(dirnum):
        initdata = np.load(dirnum+file_name, allow_pickle=True).item()
        if energy in initdata.keys():
            return initdata[energy]
    else:
        initdata = dict()

    init_x1 = lyap_init(xe, 1e-4)
    init_x2 = lyap_init(xe, 0.005)
    x_diff1 = diff_corr(init_x1)
    x_diff2 = diff_corr(init_x2)
    x_contination = Contination(x_diff1, x_diff2, energy)
    #値の保存
    initdata[energy] = x_contination
    np.save(dirnum+file_name,initdata)

    return x_contination

def main():
    t1 = time.time()
    L = Lagrange()
    E = [Phi([L[0][0], L[1][0]]) + i for i in [0.01,0.05,0.1,0.2, 0.25]]
    color = ['b', 'c', 'g', 'r', 'm']
    #図の設定
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot()

    for i in range(len(E)):
        #初期条件の計算
        L3_con = Con_init(L[0][0], E[i])
        #積分計算
        y1 = Lyapunov(L3_con)
        ax.plot(y1[0], y1[1], color=color[i], label='E={:.8f}'.format(E[i]))
    ax.plot(L[0][0],L[1][0],'k.')
    ax.set_xlabel('$\it{x}$',fontsize=25,fontstyle='italic')
    ax.set_ylabel('y',fontsize=25)
    fig.gca().set_aspect('equal')
    plt.tight_layout()

    #グラフの保存
    fig.savefig(dirgraph+"Contination.png")

    t2 = time.time()
    print(f'経過時間:{t2-t1}s')
    plt.tight_layout
    plt.show()

if __name__=='__main__':
    main()