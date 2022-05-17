import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp 

#初期条件の計算
def lyap_init(mu, Ax, xe):
    mub = mu/(abs(xe - 1 + mu)**3) + (1 - mu)/(abs(xe + mu)**3)
    nu = np.sqrt(-0.5*(mub - 2 - np.sqrt(9*mub**2 - 8*mub)))
    tau = -(nu**2 + 2*mub + 1)/(2*nu)
    x0 = xe - Ax
    vy0 = -Ax*nu*tau
    print(vy0)
    return np.array([x0,0,0,vy0])
    


#エネルギー計算
def Energy(mu,x):
    r1 = np.sqrt((x[0]+mu)**2+x[1]**2)
    r2 = np.sqrt((x[0]-1+mu)**2+x[1]**2)
    #エネルギー
    E = (x[2]**2+x[3]**2)/2-(x[0]**2+x[1]**2)/2 -(1-mu)/r1 -mu/r2 -mu*(1-mu)/2
    return E

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
    N = 100000
    tlim = np.linspace(0,100,N)
    line = solve_ivp(func, [0,100], x, t_eval=tlim, events=event_y0pm,rtol = 1.e-13, atol = 1.e-13)
    line_y = line.y
    while abs(line_y[1][-1])>1e-8:
        x = np.array([i[-1] for i in line.y])
        N = N*10
        tlim = np.linspace(0,1000000/N,N)
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
        tlim = np.linspace(0,1000000/N,N)
        line = solve_ivp(func, [0,100], x, t_eval=tlim, events=event_y0mp,rtol = 1.e-13, atol = 1.e-13)
        line_y = np.concatenate([line_y,line.y],1)
    
    return line_y

#微分補正
def diff_corr(mu,x):
    x_fin = orbit(x)
    if abs(x_fin[2][-1]) <1e-11:
        return x
    else:
        r1 = np.sqrt((x_fin[0][-1]+mu)**2+x_fin[1][-1]**2)
        r2 = np.sqrt((x_fin[0][-1]-1+mu)**2+x_fin[1][-1]**2)
        Ux = -x_fin[0][-1]+(1 - mu)*(x_fin[0][-1] + mu)/r1**3+mu*(x_fin[0][-1]-1 +mu)/r2**3
        dvy0 = ((x_fin[15][-1] -(2*x_fin[3][-1] - Ux)*x_fin[11][-1]/x_fin[3][-1])**(-1))*x_fin[2][-1]
        x_new = np.array([0,0,0,-dvy0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        return diff_corr(mu, x+x_new)

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
        x2 = diff_corr(mu, x1+delta)
        return Contination(mu, x1, x2, energy)
    else:
        x1 = x2
        x2 = diff_corr(mu, x1+delta)
        return Contination(mu, x1, x2, energy)

if __name__=='__main__':
    mu = 0.01215
    i = 0
    x_a = 1.0
    while True:
        x1 = x_a - (x_a**5 + (3-mu)*(x_a**4) + (3-2*mu)*(x_a**3) - mu*(x_a**2) -(2*mu)*x_a - mu)\
            /(5*(x_a**4) + 4*(3-mu)*(x_a**3) + 3*(3-2*mu)*(x_a**2) - 2*mu*(x_a) -2*mu)

        if np.abs(x1 - x_a) < 10e-6:
            break

        x_a = x1
    x_L2 = 1 - mu + x_a
   
    #L2周りの初期条件
    init_x1 = np.append(lyap_init(mu, 0.002, x_L2),np.eye(4).reshape(1,16))
    init_x2 = np.append(lyap_init(mu, 0.005, x_L2),np.eye(4).reshape(1,16))
    x_diff1 = diff_corr(mu,init_x1)
    x_diff2 = diff_corr(mu,init_x2)
    x_con = Contination(mu,x_diff1, x_diff2, -1.57)
    
    y = Lyapunov(x_con)
    print(y[0][0],y[1][0],y[2][0],y[3][0])
    print(y[0][-1],y[1][-1],y[2][-1],y[3][-1])
    plt.plot(y[0],y[1])
    plt.plot(x_L2,0,'kx')
    plt.xlabel('$\it{x}$',fontsize=25,fontstyle='italic')
    plt.ylabel('y',fontsize=25)
    plt.gca().set_aspect('equal')
    plt.show()
    