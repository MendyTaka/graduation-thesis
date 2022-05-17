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

#数値計算,半周分
def orbit(x):
    #数値計算
    tlim = np.linspace(0,100,950000)
    line = solve_ivp(func, [0,100], x, t_eval=tlim, events=event_y0pm,rtol = 1.e-13, atol = 1.e-13)
    return line.y

#数値計算,半周分
def orbit2(x):
    #数値計算
    tlim = np.linspace(0,100,950000)
    line = solve_ivp(func, [0,100], x, t_eval=tlim, events=event_y0mp,rtol = 1.e-13, atol = 1.e-13)
    return line.y

#微分補正
def diff_corr(mu,x):
    x_fin = orbit(x)
    print(x_fin[2][-1])
    if abs(x_fin[2][-1]) <=1e-11:
        return x
    else:
        r1 = np.sqrt((x_fin[0][-1]+mu)**2+x_fin[1][-1]**2)
        r2 = np.sqrt((x_fin[0][-1]-1+mu)**2+x_fin[1][-1]**2)
        Ux = -x_fin[0][-1]+(1 - mu)*(x_fin[0][-1] + mu)/r1**3+mu*(x_fin[0][-1]-1 +mu)/r2**3
        dvy0 = ((x_fin[15][-1] -(2*x_fin[3][-1] - Ux)*x_fin[11][-1]/x_fin[3][-1])**(-1))*x_fin[2][-1]
        x_new = np.array([0,0,0,-dvy0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        return diff_corr(mu, x+x_new)


if __name__=='__main__':
    mu = 0.01215
    L1 = 0.8369180073169304
    L2 = 1.1556799130947353
    L3 = -1.0050624018204988
    #L1周りの初期条件
    init_x = np.append(np.array([0.8359180073169304,0,0,0.008372241411491491]),np.eye(4).reshape(1,16))
    x_diff = diff_corr(mu,init_x)

    y = orbit2(x_diff)
    print(y)
    plt.plot(y[0],y[1])
    plt.plot(L1,0,'k.')
    plt.xlim(0.83,0.8390)
    plt.ylim(-0.0040,0.0040)
    plt.xlabel('$\it{x}$',fontsize=25,fontstyle='italic')
    plt.ylabel('y',fontsize=25)
    plt.gca().set_aspect('equal')
    plt.show()