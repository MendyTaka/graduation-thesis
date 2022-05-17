import os
import numpy as np
from scipy.integrate import solve_ivp 

#質量比
mu = 1.215e-2 #地球-月
#mu=9.537e-4 #太陽-木星
#mu=3.036e-6 #太陽-(地球＋月)
#mu=1.667e-8 #火星-phobos
#mu=4.704e-5 #木星-Io
#mu=2.528e-5 #木星-Europa
#mu=7.804e-5 #木星-Ganymede
#mu=5.667e-5 #木星-Callisto
#mu=6.723e-8 #Saturn-Mimas
#mu=2.366e-4 #Saturn-Titan
#mu=2.089e-4 #Neptune-Triton
#mu=1.097e-1 #Pluto-Charon

if abs(mu-1.215e-2)<1e-5:
    dir = 'earth_moon/'
elif abs(mu-9.537e-4)<1e-7:
    dir = 'sun_jupiter/'
elif abs(mu-3.036e-6)<1e-9:
    dir = 'sun_earth/'
elif abs(mu-1.667e-8)<1e-11:
    dir = 'mars_phobos/'
elif abs(mu-4.704e-5)<1e-8:
    dir = 'jupiter_io/'
elif abs(mu-2.528e-5)<1e-8:
    dir = 'jupiter_europa/'
elif abs(mu-7.804e-5)<1e-8:
    dir = 'jupiter_ganymede/'
elif abs(mu-5.667e-5)<1e-8:
    dir = 'jupiter_callisto/'
elif abs(mu-6.723e-8)<1e-11:
    dir = 'saturn_mimas/'
elif abs(mu-2.366e-4)<1e-7:
    dir = 'saturn_titan/'
elif abs(mu-2.089e-4)<1e-7:
    dir = 'neptune_triton/'
elif abs(mu-1.097e-1)<1e-4:
    dir = 'pluto_charon/'

dirnum = os.path.dirname(__file__)+"/data/"+dir
os.makedirs(dirnum, exist_ok=True)

dirgraph = os.path.dirname(__file__)+"/image/"+dir
os.makedirs(dirgraph, exist_ok=True)


#ライブラリーを使う方法
def Lagrange_lib(mu):
    #L1のx座標を求める
    L1_x=np.roots([1, -3+mu, 3-2*mu, -mu, 2*mu, -mu])
    #虚数解を取り除く
    for i in L1_x:
        if i.imag == 0:
            L1 = float(1-mu-i.real)
            break
    
    #L2のx座標を求める
    L2_x=np.roots([1, 3-mu, 3-2*mu, -mu, -2*mu, -mu])
    for i in L2_x:
        if i.imag == 0:
            L2 = float(1-mu+i.real)
            break
    
    #L3のx座標を求める
    L3_x=np.roots([1, 2+mu, 1+2*mu, -(1-mu), -2*(1-mu), -1+mu])
    for i in L3_x:
        if i.imag == 0:
            L3 = float(-mu-i.real)
            break
    
    #Lagrange点
    xpoints=np.array([L1, L2, L3, 0.5-mu, 0.5-mu])
    ypoints=np.array([0, 0, 0, np.sqrt(3)/2, -np.sqrt(3)/2])
    
    return np.array([xpoints,ypoints])

#Lagrange点
def Lagrangepoint(mu):
    #値を保存
    if 'Lagrange.npy' in os.listdir(dirnum):
        Lagrange = np.load(dirnum+'Lagrange.npy')
        return Lagrange
    x = Lagrange_lib(mu)
    np.save(dirnum+'Lagrange.npy', x)
    return x


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

#ここから積分の数値条件
#修了条件
#数値計算の終了条件　x軸を正から負に移動するとき終了
def event_y0pm(t,y):
    return y[1]-1e-16
event_y0pm.terminal = True
event_y0pm.direction = -1

#数値計算の終了条件　x軸を正から負に移動するとき終了
def event_y0mp(t,y):
    return y[1]+1e-16
event_y0mp.terminal = True
event_y0mp.direction = 1

#Tubeの終了条件 地球の左側でｘ軸を下から上の交差(U1,U4)
def event_U14mp(t,y):
    if y[0]<0:
        return y[1]+1e-16
    else:
        return 1
event_U14mp.terminal = True
event_U14mp.direction = 1

#Tubeの終了条件 地球の左側でｘ軸を上から下の交差(U1,U4)
def event_U14pm(t,y):
    if y[0]<0:
        return y[1]-1e-16
    else:
        return 1
event_U14pm.terminal = True
event_U14pm.direction = -1

#Tubeの終了条件 月の周辺を左から右に交差(U2,U3)
def event_U23mp(t,y):
    return y[0]-1+mu+1e-16
event_U23mp.terminal = True
event_U23mp.direction = 1

#Tubeの終了条件 月の周辺を右から左に交差(U2,U3)
def event_U23pm(t,y):
    return y[0]-1+mu-1e-16
event_U23pm.terminal = True
event_U23pm.direction = -1


#数値計算,半周分
def Lyapunov_half(x):
    #数値計算
    line = solve_ivp(func, [0,100], x, events=event_y0pm,rtol = 1.e-12, atol = 1.e-12)
    return line.y

#数値計算,１周分
def Lyapunov(x):
    #数値計算
    line = solve_ivp(func, [0,100], x, events=event_y0mp,rtol = 1.e-12, atol = 1.e-12)
    return line.y