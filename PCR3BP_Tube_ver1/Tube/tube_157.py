import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp 


#運動方程式と状態遷移行列の実装
def func(t, x):
    r1 = np.sqrt((x[0]+mu)**2+x[1]**2)#天体１とエネルギーの距離
    r2 = np.sqrt((x[0]-1+mu)**2+x[1]**2)#天体2とエネルギーの距離
    Ux = -x[0]+(1 - mu)*(x[0] + mu)/r1**3+mu*(x[0]-1 +mu)/r2**3
    Uy = -x[1]+(1 - mu)*x[1]/r1**3+mu*x[1]/r2**3
    Uxx = -1 + (1 - mu)/(r1**3)+mu/(r2**3)-(3*(1-mu)*(x[0]+mu)**2)/(r1**5)-(3*mu*(x[0]-1+mu)**2)/(r2**5)
    Uxy = -(3*(1-mu)*(x[0]+mu)*x[1])/(r1**5)-(3*mu*(x[0]-1+mu)*x[1])/(r2**5)
    Uyy = -1 + (1 - mu)/(r1**3)+mu/(r2**3)-(3*(1-mu)*x[1]**2)/(r1**5)-(3*mu*x[1]**2)/(r2**5)
    #運動方程式
    D = np.array([[0,0,1,0],[0,0,0,1],[0,0,0,2],[0,0,-2,0]])
    G = np.array([0,0,-Ux,-Uy])
    x1 = x[0:4]
    ans1 = np.dot(D,x1)+G
    #状態遷移関数
    Df = np.array([[0,0,1,0],[0,0,0,1],[-Uxx,-Uxy,0,2],[-Uxy,-Uyy,-2,0]])
    x2 = np.array(x[4:]).reshape([4,4])
    ans2 = np.dot(Df,x2)
    return np.append(ans1.reshape(1,4),ans2.reshape(1,16))

#数値計算の終了条件　x軸を正から負に移動するとき終了
def event_y0pm(t,y):
    return y[1]-1e-12
event_y0pm.terminal = True
event_y0pm.direction = -1

#数値計算の終了条件　x軸を負から正に移動するとき終了
def event_y0mp(t,y):
    return y[1]+1e-12
event_y0mp.terminal = True
event_y0mp.direction = 1

#Tubeの終了条件 地球の左側でｘ軸を下から上の交差
def event_Earthmp(t,y):
    if y[0]<0:
        return y[1]+1e-12
    else:
        return 1
event_Earthmp.terminal = True
event_Earthmp.direction = 1

#Tubeの終了条件　地球の左側でｘ軸を上から下の交差
def event_Earthpm(t,y):
    if y[0]<0:
        return y[1]-1e-12
    else:
        return 1
event_Earthpm.terminal = True
event_Earthpm.direction = -1

#Tubeの終了条件 月の周辺を左から右に交差
def event_Moonmp(t,y):
    return y[0]-1+mu+1e-12
event_Moonmp.terminal = True
event_Moonmp.direction = 1

#Tubeの終了条件 月の周辺を右から左に交差
def event_Moonpm(t,y):
    return y[0]-1+mu-1e-12
event_Moonpm.terminal = True
event_Moonpm.direction = -1

#数値計算,半周分
def orbit(x):
    #数値計算
    N = 100000
    tlim = np.linspace(0,100,N)
    line = solve_ivp(func, [0,100], x, t_eval=tlim, events=event_y0pm,rtol = 1.e-13, atol = 1.e-13)
    line_y = line.y
    while abs(line_y[1][-1])>1e-8:
        x = np.array([i[-1] for i in line.y])
        N = N*10
        tlim = np.linspace(0,100000/N,1001)
        line = solve_ivp(func, [0,100], x, t_eval=tlim, events=event_y0pm,rtol = 1.e-13, atol = 1.e-13)
        line_y = np.concatenate([line_y,line.y],1)
    return line_y

def Lyapunov(x):
    #数値計算
    N = 100000
    tlim = np.linspace(0,100,N)
    line = solve_ivp(func, [0,100], x, t_eval=tlim, events=event_y0mp,rtol = 1.e-13, atol = 1.e-13)
    line_y = line.y
    while abs(line_y[1][-1])>1e-8:
        x = np.array([i[-1] for i in line.y])
        N = N*10
        tlim = np.linspace(0,100000/N,1001)
        line = solve_ivp(func, [0,100], x, t_eval=tlim, events=event_y0mp,rtol = 1.e-13, atol = 1.e-13)
        line_y = np.concatenate([line_y,line.y],1)
    
    return line_y

#Tube,L1周りの計算
def Tube_L1(x_init):
    #Lyapunov軌道の計算
    x_fin = Lyapunov(x_init)
    #一周した後の状態遷移行列を計算
    matrix = np.array([x_fin[i][-1] for i in range(4,20)]).reshape(4,4)
    #固有値，固有行列の計算
    eig = np.linalg.eig(matrix)
    #固有ベクトルの設定
    eigvu = np.array([eig[1][i][0].real for i in range(4)])
    eigvs = np.array([eig[1][i][1].real for i in range(4)])

    for t in np.arange(0,100,delta_time):
        tlim = np.linspace(t,t+delta_time,2001)
        line1 = solve_ivp(func, [t,t+delta_time], x_init, t_eval=tlim, events=event_y0mp,rtol = 1.e-13, atol = 1.e-13)
        x_init = np.array([line1.y[i][-1] for i in range(20)])
        if (t>1) and (abs(x_init[1])<1e-5) and (x_init[3]>0):
            break
        #モノドロミー行列の計算
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
        tlim2 = np.linspace(0,100,100001)
        tlim3 = np.linspace(0,-100,100001)
        line3 = solve_ivp(func, [0,100], XUP, t_eval=tlim2,events=event_Earthpm,rtol = 1.e-13, atol = 1.e-13)
        line4 = solve_ivp(func, [0,100], XUM, t_eval=tlim2,events=event_Moonmp,rtol = 1.e-13, atol = 1.e-13)
        line5 = solve_ivp(func, [0,-100], XSP, t_eval=tlim3,events=event_Earthmp,rtol = 1.e-13, atol = 1.e-13)
        line6 = solve_ivp(func, [0,-100], XSM, t_eval=tlim3,events=event_Moonmp,rtol = 1.e-13, atol = 1.e-13)
        plt.plot(line3.y[0],line3.y[1],'r-')
        plt.plot(line4.y[0],line4.y[1],'r-')
        plt.plot(line5.y[0],line5.y[1],'b-')
        plt.plot(line6.y[0],line6.y[1],'b-')
        print(t)
    #モノドロミー行列
    print(x_init[4:20].reshape(4,4))
    
    return 0

#Tube,L2周りの計算
def Tube_L2(x_init):
    #Lyapunov軌道の計算
    x_fin = Lyapunov(x_init)
    #一周した後の状態遷移行列を計算
    matrix = np.array([x_fin[i][-1] for i in range(4,20)]).reshape(4,4)
    #固有値，固有行列の計算
    eig = np.linalg.eig(matrix)
    #固有ベクトルの設定
    eigvu = np.array([eig[1][i][0].real for i in range(4)])
    eigvs = np.array([eig[1][i][1].real for i in range(4)])

    for t in np.arange(0,100,delta_time):
        tlim = np.linspace(t,t+delta_time,2001)
        line1 = solve_ivp(func, [t,t+delta_time], x_init, t_eval=tlim, events=event_y0mp,rtol = 1.e-13, atol = 1.e-13)
        x_init = np.array([line1.y[i][-1] for i in range(20)])
        if (t>1) and (abs(x_init[1])<1e-5) and (x_init[3]>0):
            break
        #モノドロミー行列の計算
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
        tlim2 = np.linspace(0,100,10001)
        tlim3 = np.linspace(0,-100,10001)
        line3 = solve_ivp(func, [0,100], XUP, t_eval=tlim2,events=event_Earthmp,rtol = 1.e-13, atol = 1.e-13)
        line4 = solve_ivp(func, [0,100], XUM, t_eval=tlim2,events=event_Moonpm,rtol = 1.e-13, atol = 1.e-13)
        line5 = solve_ivp(func, [0,-100], XSP, t_eval=tlim3,events=event_Moonpm,rtol = 1.e-13, atol = 1.e-13)
        line6 = solve_ivp(func, [0,-100], XSM, t_eval=tlim3,events=event_Earthpm,rtol = 1.e-13, atol = 1.e-13)
        plt.plot(line3.y[0],line3.y[1],'r-')
        plt.plot(line4.y[0],line4.y[1],'r-')
        plt.plot(line5.y[0],line5.y[1],'b-')
        plt.plot(line6.y[0],line6.y[1],'b-')
        print(t)
    #モノドロミー行列
    print(x_init[4:20].reshape(4,4))
        
        
    
    return 0


if __name__=='__main__':
    mu = 0.01215
    L1 = 0.8369180073169304
    L2 = 1.1556799130947353
    L3 = -1.0050624018204988
    #時間幅
    delta_time = 0.1
    eps = 1e-6
    #エネルギーE=-1.57の初期値
    x_L1 = np.append(np.array([0.8112490412620138,0,0,0.259171499668709]),np.eye(4).reshape(1,16))
    x_L2 = np.append(np.array([1.09947008130107,0,0,0.27541869623794524]),np.eye(4).reshape(1,16))
    Tube_L1(x_L1)
    Tube_L2(x_L2)
    #P1(地球)
    plt.plot(-mu, 0, marker='.', markersize=20, color='k')

    #P2(月)
    plt.plot(1-mu, 0, marker='.', markersize=10, color='k')
    
    y = Lyapunov(x_L1)
    plt.plot(y[0],y[1],'g-')
    y2 = Lyapunov(x_L2)
    plt.plot(y2[0],y2[1],'g-')
    plt.xlabel('$\it{x}$',fontsize=25,fontstyle='italic')
    plt.ylabel('y',fontsize=25)
    plt.gca().set_aspect('equal')
    plt.show()
    
