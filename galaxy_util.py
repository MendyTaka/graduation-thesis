import numpy as np
import os
from scipy.integrate import solve_ivp 
from matplotlib import pyplot as plt

Omega = 1
v0 = 1
Rc = 0.1
q = 0.99

dirname = 'Omega{:.3f}_v0{:.3f}_Rc{:.3f}_q{:.3f}/'.format(Omega, v0, Rc, q)
dirnum = os.path.dirname(__file__)+"/data/"+dirname
os.makedirs(dirnum, exist_ok=True)

dirgraph = os.path.dirname(__file__)+"/image/"+dirname
os.makedirs(dirgraph, exist_ok=True)

def Lagrange():
    L12 = (v0**2)/(Omega**2) - Rc**2
    L45 = (v0**2)/(Omega**2) - (q*Rc)**2
    return np.array([[np.sqrt(L12), -np.sqrt(L12), 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, np.sqrt(L45), -np.sqrt(L45)]])


#有効ポテンシャル
Phi = lambda x: v0**2*np.log(Rc**2 + x[0]**2+(x[1]**2)/(q**2))/2 - Omega**2*(x[0]**2+x[1]**2)/2
#エネルギー
Energy = lambda x: (x[2]**2+x[3]**2)/2 +Phi(x)
#有効ポテンシャルの微分
Phix = lambda x,y: (v0**2*x)/(y**2/q**2 + Rc**2 + x**2) - Omega**2*x
Phiy = lambda x,y: (v0**2*y)/(q**2*(y**2/q**2 + Rc**2 + x**2)) - Omega**2*y
Phixx = lambda x,y: v0**2/(y**2/q**2 + Rc**2 + x**2) - Omega**2 - (2*v0**2*x**2)/(y**2/q**2 + Rc**2 + x**2)**2
Phixy = lambda x,y: -(2*v0**2*x*y)/(q**2*(y**2/q**2 + Rc**2 + x**2)**2)
Phiyy = lambda x,y: v0**2/(q**2*(y**2/q**2 + Rc**2 + x**2)) - Omega**2 - (2*v0**2*y**2)/(q**4*(y**2/q**2 + Rc**2 + x**2)**2)

#運動方程式
def func0(t, y):
    dx = y[2]
    dy = y[3]
    ddx = 2*Omega*y[3] - Phix(y[0], y[1])
    ddy = -2*Omega*y[2] - Phiy(y[0], y[1])
    return np.array([dx,dy,ddx,ddy])

#運動方程式と状態遷移行列
def func(t, y):
    dx = y[2]
    dy = y[3]
    ddx = 2*Omega*y[3] - Phix(y[0], y[1])
    ddy = -2*Omega*y[2] - Phiy(y[0], y[1])
    ans1 = np.array([dx,dy,ddx,ddy])
    #状態遷移行列
    Df = np.array([[0,0,1,0],
                [0,0,0,1],
                [-Phixx(y[0], y[1]),-Phixy(y[0], y[1]),0,2*Omega],
                [-Phixy(y[0], y[1]),-Phiyy(y[0], y[1]),-2*Omega,0]])
    y2 = np.array(y[4:]).reshape([4,4])
    ans2 = np.dot(Df,y2)
    return np.append(ans1.reshape(1,4),ans2.reshape(1,16))


#ここから積分の数値条件
#終了条件
#数値計算の終了条件　x軸を正から負に移動するとき終了
def event_y0pm(t,y):
    return y[1]-1e-16
event_y0pm.terminal = True
event_y0pm.direction = -1

#数値計算,半周分
def Lyapunov_half(x):
    #数値計算
    line = solve_ivp(func, [0,100], x, events=event_y0pm,rtol = 1.e-12, atol = 1.e-12)
    return line.y

#数値計算の終了条件　x軸を負から正に移動するとき終了
def event_y0mp(t,y):
    return y[1]+1e-16
event_y0mp.terminal = True
event_y0mp.direction = 1

#数値計算,１周分
def Lyapunov(x):
    #数値計算
    line = solve_ivp(func, [0,200], x, events=event_y0mp,rtol = 1.e-12, atol = 1.e-12)
    return line.y

def event_U12pm(t,y):
    return y[0]-1e-16
event_U12pm.terminal = True
event_U12pm.direction = -1

def event_U12mp(t,y):
    return y[0]+1e-16
event_U12mp.terminal = True
event_U12mp.direction = 1

def event_U3pm(t,y):
    if y[0]<0:
        return y[1]-1e-16
    else:
        return 1
event_U3pm.terminal = True
event_U3pm.direction = -1

def event_U3mp(t,y):
    if y[0]<0:
        return y[1]+1e-16
    else:
        return -1
event_U3mp.terminal = True
event_U3mp.direction = 1

def event_U4pm(t,y):
    if y[0]>0:
        return y[1]-1e-16
    else:
        return 1
event_U4pm.terminal = True
event_U4pm.direction = -1

def event_U4mp(t,y):
    if y[0]>0:
        return y[1]+1e-16
    else:
        return -1
event_U4mp.terminal = True
event_U4mp.direction = 1