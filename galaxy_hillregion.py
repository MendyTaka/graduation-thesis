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
    Phi,
    Lagrange
)

#hill領域の計算
def hillregion(E,N=4001):
    keyname = 'E{:.8f}_N{:d}'.format(E,N)
    if 'hillregion.npy' in os.listdir(dirnum):
        hilldata = np.load(dirnum+'hillregion.npy', allow_pickle=True).item()
        if keyname in hilldata.keys():
            return hilldata[keyname]
    else:
        hilldata = dict()

    x = np.linspace(-3, 3, N)
    y = np.linspace(-3, 3, N)
    z = np.zeros((2,N*N))
    k = 0
    for i in x:
        for j in y:
            if Phi([i,j]) > E:
                z[0][k] = i
                z[1][k] = j
                k += 1
    xpoint = z[0][:k]
    ypoint = z[1][:k]
    hilldata[keyname] = np.array([xpoint, ypoint])
    np.save(dirnum+'hillregion.npy', hilldata)

    return hilldata[keyname]

#等高線
def main():
    t1 = time.time()
    L = Lagrange()
    #パラメータの計算
    E =-0.49  #-2.3025850929940455,-0.495, -0.2736564486857903
    N=1001
    #禁止領域の計算
    z = hillregion(E, N)

    #図示
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    ax.scatter(z[0],z[1],c='#a9a9a9',marker='s')
    #Lagrange点
    ax.plot(L[0],L[1],'rx')
    ax.set_title('E={:.8f}'.format(E))
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    ax.set_xlabel('$\it{x}$',fontsize=10,fontstyle='italic')
    ax.set_ylabel('y',fontsize=10)
    fig.gca().set_aspect('equal')
    plt.tight_layout()

    #値を保存するための準備
    dirhill = dirgraph+'hill/'
    os.makedirs(dirhill, exist_ok=True)
    fig.savefig(dirhill+"hillregion(E{:.8f}_N{:d}).png".format(E,N))
    t2 = time.time()
    print('経過時間:'+str(t2-t1)+'s')
    plt.show()


    
