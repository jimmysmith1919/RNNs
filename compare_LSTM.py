import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt

##Vanilla LSTM##
##################################
def i_f_o_gates(W, b, u, h):
    x = np.concatenate((u,h))
    return expit(W @ x +b)
def g_gate(W,b,u,h):
    x = np.concatenate((u,h))
    return np.tanh(W @ x + b)

def LSTM_step(u,h,c,Wi,Wf,Wo,Wg,bi,bf,bo,bg):
    i = i_f_o_gates(Wi,bi,u,h)
    f = i_f_o_gates(Wf,bf,u,h)
    o = i_f_o_gates(Wo,bo,u,h)
    g = g_gate(Wg,bg,u,h)

    c = f*c+i*g
    h = o*np.tanh(c)
    return i,f,o,g,c,h
####################################


##Stochastic LSTM##
###################################
def sample_z(gate):
    return  np.random.binomial(1, gate)

def break_stick(alpha, tau, x):
    pi = np.zeros((len(x), 3))
    pi[:,0] = expit((-alpha-x)/tau) #x<-alpha
    pi[:,1] = (1-pi[:,0])*expit((x-alpha)/tau) #x>alpha
    pi[:,2] = expit((alpha+x)/tau)*expit((-x+alpha)/tau) #-alpha<=x<=alpha
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

def get_mu(y, r, alpha):
    mu = np.zeros(len(y))
    for i in range(0,len(y)):
        mu[i] = tanh_approx(r[i],alpha, y[i])
    return mu

def stoch_LSTM_step(u,h,c,Wi,Wf,Wo,Wg,bi,bf,bo,bg,alpha1,alpha2,tau1,tau2,
                    sigma1,sigma2,sigma3):
    x = np.concatenate((u,h))

    #Sample z's for i, f and o gates
    i = i_f_o_gates(Wi,bi,u,h)
    f = i_f_o_gates(Wf,bf,u,h)
    o = i_f_o_gates(Wo,bo,u,h)
    zi = sample_z(i)
    zf = sample_z(f)
    zo = sample_z(o)

    #Sample g gate
    g_lin = Wg @ x + bg
    pi_g=break_stick(alpha1, tau1, g_lin)
    rg = sample_r(pi_g)
    mu_g = get_mu(h, rg, alpha1)
    g = np.random.normal(mu_g, sigma1)
    #sample c
    c = np.random.normal(zf*c+zi*g, sigma2)

    #Sample h
    pi_h=break_stick(alpha2, tau2, c)
    rh = sample_r(pi_h)
    mu_h = get_mu(c, rh, alpha2)
    h =  np.random.normal(zo*mu_h, sigma3)
    return c, h, zi, zf, zo, g
############################################3
'''
##Dimensions##
D1=5 #h_t and c_t dimension
D2 = 2 #u_t dimension

##Random Weights##
Wi = np.random.randn(D1,D1+D2)
Wf = np.random.randn(D1,D1+D2)
Wo = np.random.randn(D1,D1+D2)
Wg = np.random.randn(D1,D1+D2)

bi =np.random.randn(D1)
bf =np.random.randn(D1)
bo =np.random.randn(D1)
bg =np.random.randn(D1)

##Initialize##
c = np.ones(D1) #Initial cell state
h = np.ones(D1) #Initial hidden state
u = 0*np.ones(D2) #initial inputs

T = 10 #Timesteps

##Generate Vanilla LSTM 

for t in range(0,T):
    i,f,o,g,c,h =LSTM_step(u, h, c,Wi,Wf,Wo,Wg,bi,bf,bo,bg)


#Stochastic LSTM hyperparameters
alpha1 = 1.5
alpha2 = 1.5
tau1 = 1
tau2 =1
sigma1 = .1
sigma2 = .1
sigma3 = .1

#Generate Stochastic LSTM
c_vec = []
h_vec = []
i_vec = []
f_vec = []
o_vec = []
g_vec = []
for t in range(0,T):
    c,h, zi, zf, zo, g = stoch_LSTM_step(u,h,c,Wi,Wf,Wo,Wg,bi,bf,bo,bg)
    c_vec.append(c)
    h_vec.append(h)
    i_vec.append(zi)
    f_vec.append(zf)
    o_vec.append(zo)
    g_vec.append(g)
    

t_vec = np.arange(0,T,1)
print(c_vec[0])
plt.plot(t_vec, c_vec, 'c')
plt.show()
'''

