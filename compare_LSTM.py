import numpy as np
from scipy.special import expit

D1=10 #h_t and c_t dimension
D2 = 5 #u_t dimension

Wi = np.random.randn(D1,D1+D2)
Wf = np.random.randn(D1,D1+D2)
Wo = np.random.randn(D1,D1+D2)
Wg = np.random.randn(D1,D1+D2)

bi =0* np.random.randn(D1)
bf = 0*np.random.randn(D1)
bo = 0*np.random.randn(D1)
bg = 0*np.random.randn(D1)



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
    return i,f,o,g,c,h


c = np.ones(D1)
h = np.ones(D1)
u = np.zeros(D2)
T = 10
'''
for t in range(0,T):
    i,f,o,g,c,h =LSTM_step(u, h, c)
'''

def sample_z(gate):
    return  np.random.binomial(1, gate)

def break_stick(alpha, tau, x):
    pi = np.zeros((len(x), 3))
    pi[:,0] = expit((-alpha-x)/tau) #x<-alpha
    pi[:,1] = (1-pi[:,0])*expit((x-alpha)/tau) #x>alpha
    pi[:,2] = expit((alpha+x)/tau)*expit((-x+alpha)/tau)
    return pi

def sample_r(pi):
    r = np.zeros(len(pi[:,0]))
    for i in range(0,len(pi[:,0])):
        r[i] = np.argmax(np.random.multinomial(1,pi[i,:]))
    return r
    
def tanh_approx(r,alpha, x):
    if r == 0: #x<-alpha
        return -1
    elif r == 1: #x>alpha:
        return 1
    else:
        return -1+(x+alpha)/alpha

def sample_g(h, rg, sigma, alpha):
    mu_g = np.zeros(len(h))
    for i in range(0,len(h)):
        mu_g[i] = tanh_approx(rg[i],alpha, h[i])
    return np.random.normal(mu_g, sigma)

def sample_h(c, rh, sigma, alpha, zo):
    mu_h = np.zeros(len(c))
    for i in range(0,len(c)):
        mu_h[i] = tanh_approx(rh[i], alpha, c[i])
    return np.random.normal(zo*mu_h, sigma)



alpha1 = 1.5
alpha2 = 1.5
tau1 = 1
tau2 =1
sigma = 1
x = np.concatenate((u,h))
g_lin = Wg @ x + bg
pi_g=break_stick(alpha1, tau1, g_lin)
rg = sample_r(pi_g)

g = sample_g(h, rg, sigma, alpha1)
i,f,o,g,c,h = LSTM_step(u,h,c)

zi = sample_z(i)
zf = sample_z(f)
zo = sample_z(o)

c = np.random.normal(zf*c+zi*g, sigma)

pi_h=break_stick(alpha2, tau2, c)
rh = sample_r(pi_h)
h = sample_h(c, rh, sigma, alpha2, zo)
print(h)
    




