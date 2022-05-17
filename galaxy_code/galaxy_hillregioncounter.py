import numpy as np
import matplotlib.pyplot as plt
import time

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

#等高線
def main():
    t1 = time.time()
    #調べる範囲
    x = np.linspace(-1.5,1.5,501)
    y = np.linspace(-1.5,1.5,501)
    
    X,Y = np.meshgrid(x,y)
    Z = Phi([X,Y])
    L = Lagrange()
    #等高線図の出力
    fig = plt.figure()
    ax = fig.add_subplot()
    line = np.arange(-1.4,0,0.2)
    for i in range(len(line)):
        if line[i]>Phi([L[0][0],L[1][0]]):
            line = np.insert(line, i, Phi([L[0][0],L[1][0]]))
            break
    cont = ax.contour(X,Y,Z,cmap='jet', levels = line)
    fig.colorbar(cont)
    #Lagrange点
    ax.plot(L[0],L[1],'ko')
    ax.set_xlabel('$\it{x}$',fontsize=25,fontstyle='italic')
    ax.set_ylabel('y',fontsize=25)
    fig.gca().set_aspect('equal')
    ax.set_title('hill_coutour')
    plt.tight_layout()
    #グラフの保存
    fig.savefig(dirgraph+'hill_coutour.png')
    t2 = time.time()
    print(f'経過時間:{t2-t1}s')
    plt.show()


    
if __name__=="__main__":
    main()