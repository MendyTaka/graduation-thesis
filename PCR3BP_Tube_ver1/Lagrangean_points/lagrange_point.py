import numpy as np
import matplotlib.pyplot as plt

mu = 0.01215

#L2を計算
i = 0
x_a = 1.0
while True:
    x1 = x_a - (x_a**5 + (3-mu)*(x_a**4) + (3-2*mu)*(x_a**3) - mu*(x_a**2) -(2*mu)*x_a - mu)\
        /(5*(x_a**4) + 4*(3-mu)*(x_a**3) + 3*(3-2*mu)*(x_a**2) - 2*mu*(x_a) -2*mu)

    if np.abs(x1 - x_a) < 10e-6:
        break

    x_a = x1
x_L2 = 1 - mu + x_a

#L1を計算
x_b = 1.0
while True:
    x2 = x_b - (x_b**5 - (3-mu)*(x_b**4) + (3-2*mu)*(x_b**3) - mu*(x_b**2) + (2*mu)*x_b - mu)\
        /(5*(x_b**4) - 4*(3-mu)*(x_b**3) + 3*(3-2*mu)*(x_b**2) - 2*mu*(x_b) + 2*mu)

    if np.abs(x2 - x_b) < 10e-6:
        break

    x_b = x2
x_L1 = 1 - mu -x_b

#L3を計算
x_c = 0.1
while True:
    x3 = x_c - (x_c**5 - (7+mu)*(x_c**4) + (19+6*mu)*(x_c**3) - (24+13*mu)*(x_c**2) + (12+14*mu)*x_c - 7*mu)\
        /(5*(x_b**4) - 4*(7+mu)*(x_c**3) + 3*(19+6*mu)*(x_c**2) - 2*(24+13*mu)*(x_c) + (12+14*mu))

    if np.abs(x3 - x_c) < 10e-6:
        break

    x_c = x3
x_L3 = -1 - mu + x_c

#P1(地球)
plt.plot(-mu, 0, marker='.', markersize=20, color='b')

#P2(月)
plt.plot(1-mu, 0, marker='.', markersize=10, color='y')

#L1
plt.plot(x_L1, 0, marker='.', markersize=5, color='k')
plt.text(x_L1-0.05, -0.2, '$L_1$',fontsize=15)

#L2
plt.plot(x_L2, 0, marker='.', markersize=5, color='k')
plt.text(x_L2-0.05, -0.2, '$L_2$',fontsize=15)

#L3
plt.plot(x_L3, 0, marker='.', markersize=5, color='k')
plt.text(x_L3-0.05, -0.2, '$L_3$',fontsize=15)

#L4
plt.plot(0.5-mu, np.sqrt(3)/2, marker='.', markersize=5, color='k')
plt.text(0.5-mu-0.05, np.sqrt(3)/2-0.2, '$L_4$',fontsize=15)

#L5
plt.plot(0.5-mu, -np.sqrt(3)/2, marker='.', markersize=5, color='k')
plt.text(0.5-mu-0.05, -np.sqrt(3)/2-0.2, '$L_5$',fontsize=15)

plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
plt.xlabel('$\it{x}$',fontsize=20,fontstyle='italic')
plt.ylabel('y',fontsize=20)
plt.gca().set_aspect('equal')
#plt.title('地球-月系Lagrange点')
plt.show()

print(x_L1)