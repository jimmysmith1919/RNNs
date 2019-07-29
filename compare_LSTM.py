import numpy as np
from scipy.special import expit

D1=2 #h_t and c_t dimension
D2 = 1 #u_t dimension

Wi = np.random.randn(D1,D1+D2)
Wf = np.random.randn(D1,D1+D2)
Wo = np.random.randn(D1,D1+D2)
Wg = np.random.randn(D1,D1+D2)

bi =0* np.random.randn(D1)
bf = 0*np.random.randn(D1)
bo = 0*np.random.randn(D1)
bg = 0*np.random.randn(D1)


'''
def i_f_o_gates(W, b, u, h):
    x = np.concatenate((u,h))
    return expit(W @ x +b)
def g_gate(W,b,u,h):
    x = np.concatenate((u,h))
    return np.tanh(W @ x + b)

def LSTM_step(u,h,c):
    i = i_f_o_gates(Wi,bi,u,h)
    f = i_f_o_gates(Wf,bf,u,h)
    o = i_f_o_gates(Wo,bo,u,h)
    g = g_gate(Wg,bg,u,h)

    c = f*c+i*g
    h = o*np.tanh(c)
    return h, c


c = np.ones(D1)
h = np.ones(D1)
u = np.zeros(D2)
T = 10

for t in range(0,T):
    h,c =LSTM_step(u, h, c)
'''

