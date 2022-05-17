import numpy as np
from matplotlib import pyplot as plt
import os,time

import Tube

#共通変数,関数
from util import (
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
    Lyapunov,
    event_U3mp,
    event_U12mp,
    event_U12pm,
    event_U3pm,
    event_U4mp,
    event_U4pm
)


#Tube,L1周りの計算
def Poincare(energy, N=100, eps=1e-6):
    #データがあるかの確認
    keyname = 'E{:.3f}_N{:d}_eps{:.1e}'.format(energy, N, eps)
    if 'Poincare.npy' in os.listdir(dirnum):
        poincare = np.load(dirnum+'Poincare.npy', allow_pickle=True).item()
        if keyname in poincare.keys():
            return poincare[keyname]
            
        else:
            poincare[keyname] = dict()
    else:
        poincare = {keyname:dict()}
    #Tubeのデータがあるか確認
    PoinL1 = Tube.Tube_L1(energy, N, eps)
    PoinL2 = Tube.Tube_L2(energy, N, eps)
    
    tube_L1_name = ['U1_unstable', 'U2_stable', 'U3_stable', 'U3_unstable']
    for L1_name in tube_L1_name:
        poincare[keyname][L1_name]=np.zeros([20, N])
        for i in range(20):
            for j in range(N):
                poincare[keyname][L1_name][i][j] = PoinL1[L1_name][j][i][-1]

    tube_L2_name = ['U4_stable', 'U4_unstable', 'U2_unstable', 'U1_stable']
    for L2_name in tube_L2_name:
        poincare[keyname][L2_name]=np.zeros([20, N])
        for i in range(20):
            for j in range(N):
                poincare[keyname][L2_name][i][j] = PoinL2[L2_name][j][i][-1]

    np.save(dirnum+'Poincare.npy', poincare)
    return poincare[keyname]


def main():
    t1 = time.time()
    E = -0.49
    N=100
    #ポアンカレ断面の計算
    poincare = Poincare(E,N)
    
    #図
    #U1
    fig1 = plt.figure(figsize=(6,5))
    ax1 = fig1.add_subplot()
    ax1.scatter(poincare['U4_stable'][1], poincare['U4_stable'][3],c='b',s=5,label='stable')
    ax1.scatter(poincare['U3_unstable'][1], poincare['U3_unstable'][3],c='r',s=5, label='unstable')
    #図の設定
    ax1.set_xlabel('$\it{y}$',fontsize=25,fontstyle='italic')
    ax1.set_ylabel('$v_y$',fontsize=25)
    ax1.legend()
    plt.tight_layout()
    fig1.savefig(dirgraph+'Poincare_U1(E={:.3f}).png'.format(E))

    #U2
    fig2 = plt.figure(figsize=(6,5))
    ax2 = fig2.add_subplot()
    ax2.scatter(poincare['U3_stable'][1], poincare['U3_stable'][3],c='b',s=5,label='stable')
    ax2.scatter(poincare['U4_unstable'][1], poincare['U4_unstable'][3],c='r',s=5, label='unstable')
    #図の設定
    ax2.set_xlabel('$\it{y}$',fontsize=25,fontstyle='italic')
    ax2.set_ylabel('$v_y$',fontsize=25)
    ax2.legend()
    plt.tight_layout()
    fig2.savefig(dirgraph+'Poincare_U2(E={:.3f}).png'.format(E))

    #U3
    fig3 = plt.figure(figsize=(6,5))
    ax3 = fig3.add_subplot()
    ax3.scatter(poincare['U2_stable'][1], poincare['U2_stable'][3],c='b',s=5,label='stable')
    ax3.scatter(poincare['U2_unstable'][1], poincare['U2_unstable'][3],c='r',s=5, label='unstable')
    #図の設定
    ax3.set_xlabel('$\it{y}$',fontsize=25,fontstyle='italic')
    ax3.set_ylabel('$v_y$',fontsize=25)
    ax3.legend()
    plt.tight_layout()
    fig3.savefig(dirgraph+'Poincare_U3(E={:.3f}).png'.format(E))

    #U4
    fig4 = plt.figure(figsize=(6,5))
    ax4 = fig4.add_subplot()
    ax4.scatter(poincare['U1_stable'][1], poincare['U1_stable'][3],c='b',s=5,label='stable')
    ax4.scatter(poincare['U1_unstable'][1], poincare['U1_unstable'][3],c='r',s=5, label='unstable')
    #図の設定
    ax4.set_xlabel('$\it{y}$',fontsize=25,fontstyle='italic')
    ax4.set_ylabel('$v_y$',fontsize=25)
    ax4.legend()
    plt.tight_layout()
    fig4.savefig(dirgraph+'Poincare_U4(E={:.3f}).png'.format(E))

    t2 = time.time()
    print(f'経過時間:{t2-t1}s')
    plt.show()


if __name__=='__main__':
    main()