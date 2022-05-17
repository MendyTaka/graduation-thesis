import numpy as np
import matplotlib.pyplot as plt
import os
import time
import Lagrange

#共通変数
from util import (
    mu,
    dirnum,
    dirgraph
)

#ゼロ速度のエネルギー計算
def Energy0(mu,x,y):
    r1 = np.sqrt((x+mu)**2+y**2)
    r2 = np.sqrt((x-1+mu)**2+y**2)
    #有効ポテンシャル
    U = -(x**2+y**2)/2 -(1-mu)/r1 -mu/r2 -mu*(1-mu)/2
    return U

#hill領域の図示
def hillregion(mu,x,y,E):
    #hillrigionのデータ構造{E1:{'x':list,'y':list,'hill:'list},E2:..}
    if 'hillregion.npy' in os.listdir(dirnum):
        hilldata = np.load(dirnum+'hillregion.npy', allow_pickle=True).item()
        if E in hilldata.keys():
            if all(hilldata[E]['x']==x) and all(hilldata[E]['y']==y):
                return hilldata[E]['hill']
    else:
        hilldata = dict()

    z = np.zeros((2,len(x)*len(y)))
    k = 0
    for i in x:
        for j in y:
            if Energy0(mu,i,j) > E:
                z[0][k] = i
                z[1][k] = j
                k += 1
    xpoint = z[0][:k]
    ypoint = z[1][:k]
    hill = np.array([xpoint, ypoint]) 
    #値の保存
    hilldata[E]={'x':x, 'y':y, 'hill':hill}
    np.save(dirnum+'hillregion.npy',hilldata)
    
    return hill
     

def main():
    t1 = time.time()
    #探索範囲を確定
    x = np.linspace(-2,2,401)
    y = np.linspace(-2,2,401)
    E = -1.57#[-1.610, -1.595, -1.570, -1.540, -1.510, -1.50]
    #hillを計算
    z = hillregion(mu, x, y, E)

    #図示
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    ax.scatter(z[0],z[1],c='#a9a9a9',marker='s')
    #P1(地球)
    ax.plot(-mu, 0, marker='.', markersize=20, color='b')
    #P2(月)
    ax.plot(1-mu, 0, marker='.', markersize=10, color='y')
    #Lagrange点
    ax.plot(Lagrange.Lagrangepoint(mu)[0],Lagrange.Lagrangepoint(mu)[1],'rx')
    ax.set_title('E={:.3f}'.format(E))
    ax.set_xlim(x[0],x[-1])
    ax.set_ylim(y[0],y[-1])
    ax.set_xlabel('$\it{x}$',fontsize=10,fontstyle='italic')
    ax.set_ylabel('y',fontsize=10)
    fig.gca().set_aspect('equal')
    plt.tight_layout()

    #値を保存するための準備
    fig.savefig(dirgraph+"hillregion(E={:.3f}).png".format(E))
    t2 = time.time()
    print('経過時間:'+str(t2-t1)+'s')
    plt.show()

if __name__=="__main__":
    main()