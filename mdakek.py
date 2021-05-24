import numpy as np
from numpy import matmul as mm
from numpy.lib.shape_base import tile
from numpy.lib.twodim_base import eye
from scipy import integrate
import matplotlib.pyplot as plt
import math
from math import gamma, pi, sin, cos
from copy import copy


def kf(X=np.array([[]]),
       U=np.array([[]]),
       F=np.array([[]]),
       G=np.array([[]]),
       P=np.array([[]]),
       Q=np.array([[]]),
       Gamma=np.array([[]]),
       Z=np.array([[]]),
       H=np.array([[]]),
       R=np.array([[]])):
    
    X_prior = mm(F, X) + mm(G, U)
    P_prior = mm(mm(F, P), np.transpose(F)) + mm(mm(Gamma, Q), np.transpose(Gamma))
    print(mm(mm(F, P), np.transpose(F)))
    print(mm(mm(Gamma, Q), np.transpose(Gamma)))
    print(P_prior)

    y = Z - mm(H, X_prior)

    S = mm(mm(H, P_prior), np.transpose(H)) + R
    K = mm(mm(P_prior, np.transpose(H)), np.linalg.inv(S))

    X_post = X_prior + mm(K, y)
    P_post = mm((np.eye(len(X))) - mm(K, H) , P_prior)

    return X_post, P_post

t = np.linspace(0,8, 100)
delta_t = t[1] - t[0]

F = np.array([[1, 0, delta_t, 0],
              [0, 1, 0, delta_t],
              [0,0,1,0],
              [0,0,0,1]])
G = np.array([[delta_t**2, 0],
              [0, delta_t**2],
              [delta_t, 0],
              [0, delta_t]])
Gamma = np.array([[delta_t**2, 0],
                  [0, delta_t**2],
                  [delta_t, 0],
                  [0, delta_t]])

Q = np.eye(2)

def U_x(t):
    return 2.5*sin(t)
def U_y(t):
    return -2*cos(t)

def se_solve(Y, t):
    _,_,v_x,v_y = Y
    return [v_x, v_y, U_x(t), U_y(t)]
sol = integrate.odeint(se_solve, [1,1,0,0], t)

V = np.random.randn(2, len(t))*2
for i in range(len(t)):
    if i == 0: continue
    if i % 2:
        sign = 1
    else:
        sign = -1
    V[0][i] = -1 * sign
    V[1][i] = -1 * sign
    
R = np.eye(2)*4
Z = np.array([sol[:, 0]+V[0, :],
              sol[:,1]+V[1,:]])
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
X_i_1 = np.array([[1], [1], [0], [0]])
P_i_1 = np.eye(4)*0.1
t_i_1 = 0
x = [1]
y = [1]
P = [P_i_1]

for t_i in range(len(t))[1:]:
    if t_i == 2: break
    print("Seep:", t_i)
    U_i = np.array([[U_x(t[t_i])],
                    [U_y(t[t_i])]])
    Z_i = np.array([[Z[0, t_i]],
                    [Z[1, t_i]]])
    X_i, P_i = kf(X_i_1, U_i, F, G, P_i_1, Q, Gamma, Z_i, H, R)
    # print(X_i)
    # print(P_i)

    x.append(X_i[0][0])
    y.append(X_i[1][0])

    t_i_1 = t_i
    X_i_1 = copy(X_i)
    P_i_1 = copy(P_i)


plt.plot(sol[:, 0], sol[:,1], Z[0,:], Z[1,:], '--', x,y)
plt.show()

 