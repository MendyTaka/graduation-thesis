import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import os,time
import hillregion, Lyapunov_con

#共通変数,関数
from util import (
    mu,
    dirnum,
    dirgraph,
    Lagrangepoint,
    func,
    event_y0pm,
    event_y0mp,
    event_U14pm,
    event_U14mp,
    event_U23mp,
    event_U23pm,
    Lyapunov_half,
    Lyapunov
)


#Tube,L1周りの計算
def Tube_L1(energy, N=100, eps=1e-6):
    dirL1 = dirnum+'TubeL1/'
    os.makedirs(dirL1, exist_ok=True)
    filename = 'E{:.3f}_N{:d}_eps{:.1e}'.format(energy, N, eps)
    if 'TubeL1_check.npy' in os.listdir(dirL1):
        TubeL1_check = np.load(dirL1+'TubeL1_check.npy')
        if filename in TubeL1_check:
            d = dict()
            U1_stable = np.load(dirL1+'U1_stable_'+filename+'.npy', allow_pickle=True).item()
            U1_unstable = np.load(dirL1+'U1_unstable_'+filename+'.npy', allow_pickle=True).item()
            U2_unstable = np.load(dirL1+'U2_unstable_'+filename+'.npy', allow_pickle=True).item()
            U3_stable = np.load(dirL1+'U3_stable_'+filename+'.npy', allow_pickle=True).item()
            return {'U1_stable':U1_stable['U1_stable'],'U1_unstable':U1_unstable['U1_unstable'],'U2_unstable':U2_unstable['U2_unstable'],'U3_stable':U3_stable['U3_stable']}
        
    else:
        TubeL1_check = np.array([])
        

    #初期値
    L = Lagrangepoint(mu)
    x0 = Lyapunov_con.Con_init(mu, L[0][0], energy)
    #Lyapunov軌道の計算
    x_fin = Lyapunov(x0)
    #一周した後の状態遷移行列を計算
    matrix = np.array([x_fin[i][-1] for i in range(4,20)]).reshape(4,4)
    #固有値，固有行列の計算
    eig = np.linalg.eig(matrix)
    #固有ベクトルの設定
    eigvu = np.array([eig[1][i][0].real for i in range(4)])
    eigvs = np.array([eig[1][i][1].real for i in range(4)])
    orbit_num = len(x_fin[0])/N
    TubeL1 = dict()
    TubeL1['U1_stable'] = [0 for _ in range(N)]
    TubeL1['U1_unstable'] = [0 for _ in range(N)]
    TubeL1['U2_unstable'] = [0 for _ in range(N)]
    TubeL1['U3_stable'] = [0 for _ in range(N)]

    for i in range(N):
        #モノドロミー行列の計算
        x_init = np.array([x_fin[j][int(i*orbit_num)] for j in range(20)])
        Monodromy = x_init[4:20].reshape(4,4)
        #安定方向，不安定方向の計算
        V2s = np.dot(Monodromy,eigvs)
        V2u = np.dot(Monodromy,eigvu)
        YS = V2s/ np.linalg.norm(V2s)
        YU = V2u/ np.linalg.norm(V2u)
        #初期値の計算
        XUP = np.concatenate([x_init[0:4]+eps*YU, x_init[4:20]])
        XUM = np.concatenate([x_init[0:4]-eps*YU, x_init[4:20]])
        XSP = np.concatenate([x_init[0:4]+eps*YS, x_init[4:20]])
        XSM = np.concatenate([x_init[0:4]-eps*YS, x_init[4:20]])
        line3 = solve_ivp(func, [0,200], XUP,events=event_U14pm,rtol = 1.e-12, atol = 1.e-12)
        line4 = solve_ivp(func, [0,200], XUM,events=event_U23mp,rtol = 1.e-12, atol = 1.e-12)
        line5 = solve_ivp(func, [0,-200], XSP,events=event_U14mp,rtol = 1.e-12, atol = 1.e-12)
        line6 = solve_ivp(func, [0,-200], XSM,events=event_U23mp,rtol = 1.e-12, atol = 1.e-12)
        TubeL1['U1_stable'][i] = line5.y
        TubeL1['U1_unstable'][i] = line3.y
        TubeL1['U2_unstable'][i] = line4.y
        TubeL1['U3_stable'][i] = line6.y
        print(f'L1:{100*(i+1)//N}%')
    
    U1_stable = {'U1_stable':TubeL1['U1_stable']}
    U1_unstable = {'U1_unstable':TubeL1['U1_unstable']}
    U2_unstable = {'U2_unstable':TubeL1['U2_unstable']}
    U3_stable = {'U3_stable':TubeL1['U3_stable']}

    np.save(dirL1+'TubeL1_check.npy',np.append(TubeL1_check, filename))
    np.save(dirL1+'U1_stable_{}.npy'.format(filename), U1_stable)
    np.save(dirL1+'U1_unstable_'+filename+'.npy', U1_unstable)
    np.save(dirL1+'U2_unstable_'+filename+'.npy', U2_unstable)
    np.save(dirL1+'U3_stable_'+filename+'.npy', U3_stable)

    return TubeL1

#Tube,L2周りの計算
def Tube_L2(energy, N=100, eps=1e-6):
    dirL2 = dirnum+'TubeL2/'
    os.makedirs(dirL2, exist_ok=True)
    filename = 'E{:.3f}_N{:d}_eps{:.1e}'.format(energy, N, eps)
    if 'TubeL2_check.npy' in os.listdir(dirL2):
        TubeL2_check = np.load(dirL2+'TubeL2_check.npy')
        if filename in TubeL2_check:
            d = dict()
            U4_stable = np.load(dirL2+'U4_stable_'+filename+'.npy', allow_pickle=True).item()
            U4_unstable = np.load(dirL2+'U4_unstable_'+filename+'.npy', allow_pickle=True).item()
            U3_unstable = np.load(dirL2+'U3_unstable_'+filename+'.npy', allow_pickle=True).item()
            U2_stable = np.load(dirL2+'U2_stable_'+filename+'.npy', allow_pickle=True).item()
            return {'U4_stable':U4_stable['U4_stable'],'U4_unstable':U4_unstable['U4_unstable'],'U3_unstable':U3_unstable['U3_unstable'],'U2_stable':U2_stable['U2_stable']}
        
    else:
        TubeL2_check = np.array([])

    #初期値
    L= Lagrangepoint(mu)
    x0 = Lyapunov_con.Con_init(mu, L[0][1], energy)
    #Lyapunov軌道の計算
    x_fin = Lyapunov(x0)
    #一周した後の状態遷移行列を計算
    matrix = np.array([x_fin[i][-1] for i in range(4,20)]).reshape(4,4)
    #固有値，固有行列の計算
    eig = np.linalg.eig(matrix)
    #固有ベクトルの設定
    eigvu = np.array([eig[1][i][0].real for i in range(4)])
    eigvs = np.array([eig[1][i][1].real for i in range(4)])
    orbit_num = len(x_fin[0])/N
    #データ保存の準備
    TubeL2 = dict()
    TubeL2['U4_stable'] = [0 for _ in range(N)]
    TubeL2['U4_unstable'] = [0 for _ in range(N)]
    TubeL2['U3_unstable'] = [0 for _ in range(N)]
    TubeL2['U2_stable'] = [0 for _ in range(N)]

    for i in range(N):
        #モノドロミー行列の計算
        x_init = np.array([x_fin[j][int(i*orbit_num)] for j in range(20)])
        Monodromy = x_init[4:20].reshape(4,4)
        #安定方向，不安定方向の計算
        V2s = np.dot(Monodromy,eigvs)
        V2u = np.dot(Monodromy,eigvu)
        YS = V2s/ np.linalg.norm(V2s)
        YU = V2u/ np.linalg.norm(V2u)
        #初期値の計算
        XUP = np.concatenate([x_init[0:4]+eps*YU, x_init[4:20]])
        XUM = np.concatenate([x_init[0:4]-eps*YU, x_init[4:20]])
        XSP = np.concatenate([x_init[0:4]+eps*YS, x_init[4:20]])
        XSM = np.concatenate([x_init[0:4]-eps*YS, x_init[4:20]])
        line3 = solve_ivp(func, [0,200], XUP, events=event_U14mp,rtol = 1.e-12, atol = 1.e-12)
        line4 = solve_ivp(func, [0,200], XUM, events=event_U23pm,rtol = 1.e-12, atol = 1.e-12)
        line5 = solve_ivp(func, [0,-200], XSP, events=event_U23pm,rtol = 1.e-12, atol = 1.e-12)
        line6 = solve_ivp(func, [0,-200], XSM, events=event_U14pm,rtol = 1.e-12, atol = 1.e-12)
        TubeL2['U2_stable'][i] = line5.y
        TubeL2['U4_unstable'][i] = line3.y
        TubeL2['U3_unstable'][i] = line4.y
        TubeL2['U4_stable'][i] = line6.y
        print(f'L2:{100*(i+1)//N}%')
        
    U4_stable = {'U4_stable':TubeL2['U4_stable']}
    U4_unstable = {'U4_unstable':TubeL2['U4_unstable']}
    U3_unstable = {'U3_unstable':TubeL2['U3_unstable']}
    U2_stable = {'U2_stable':TubeL2['U2_stable']}

    np.save(dirL2+'TubeL2_check.npy',np.append(TubeL2_check, filename))
    np.save(dirL2+'U4_stable_{}.npy'.format(filename), U4_stable)
    np.save(dirL2+'U4_unstable_'+filename+'.npy', U4_unstable)
    np.save(dirL2+'U3_unstable_'+filename+'.npy', U3_unstable)
    np.save(dirL2+'U2_stable_'+filename+'.npy', U2_stable)
    return TubeL2

def main():
    t1 = time.time()
    E = -1.57
    N=50
    L = Lagrangepoint(mu)
    #Tube計算
    x1 = Tube_L1(E,N)
    x2 = Tube_L2(E,N)
    
    #図の設定
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot()
    #禁止領域
    z = hillregion.hillregion(mu, np.linspace(-2,2,401), np.linspace(-2,2,401), E)
    ax.scatter(z[0],z[1],c='#a9a9a9',marker='s')
    #Tube
    for i in range(N):
        ax.plot(x1['U1_stable'][i][0],x1['U1_stable'][i][1],'b-')
        ax.plot(x1['U1_unstable'][i][0],x1['U1_unstable'][i][1],'r-')
        ax.plot(x1['U2_unstable'][i][0],x1['U2_unstable'][i][1],'r-')
        ax.plot(x1['U3_stable'][i][0],x1['U3_stable'][i][1],'b-')
        ax.plot(x2['U4_stable'][i][0],x2['U4_stable'][i][1],'b-')
        ax.plot(x2['U4_unstable'][i][0],x2['U4_unstable'][i][1],'r-')
        ax.plot(x2['U3_unstable'][i][0],x2['U3_unstable'][i][1],'r-')
        ax.plot(x2['U2_stable'][i][0],x2['U2_stable'][i][1],'b-')
    #Lyapunov軌道
    L1_con = Lyapunov_con.Con_init(mu, L[0][0], E)
    L2_con = Lyapunov_con.Con_init(mu, L[0][1], E)
    #積分計算
    y1 = Lyapunov(L1_con)
    y2 = Lyapunov(L2_con)
    ax.plot(y1[0], y1[1], 'g-')
    ax.plot(y2[0], y2[1], 'g-')
    ax.plot(L[0][0],L[1][0],'k.')
    ax.plot(L[0][1],L[1][1],'k.')
    ax.plot(-mu, 0, marker='.', markersize=20, color='g')
    ax.plot(1-mu, 0, marker='.', markersize=10, color='y')
    ax.set_xlabel('$\it{x}$',fontsize=25,fontstyle='italic')
    ax.set_ylabel('y',fontsize=25)
    fig.gca().set_aspect('equal')
    plt.tight_layout()
    #グラフの保存
    fig.savefig(dirgraph+'Tube'+'(E={:.3f})'.format(E)+".png")

    t2 = time.time()
    print(f'経過時間:{t2-t1}s')
    plt.show()


if __name__=='__main__':
    main()