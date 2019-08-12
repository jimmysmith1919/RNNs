import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt

##Vanilla LSTM##
##################################
def LSTM_step(u,h,c,Wi,Wf,Wo,Wg,bi,bf,bo,bg):
    x = np.concatenate((u,h))
    i = expit(Wi @ x + bi) 
    f = expit(Wf @ x + bf)
    o = expit(Wo @ x + bo)
    g = np.tanh(Wg @ x + bg)

    c = f*c+i*g
    h = o*np.tanh(c)
    return i,f,o,g,c,h
####################################


##Stochastic LSTM##
###################################
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
    i = expit(Wi @ x + bi) 
    f = expit(Wf @ x + bf)
    o = expit(Wo @ x + bo)
    zi = np.random.binomial(1,i) 
    zf = np.random.binomial(1,f) 
    zo = np.random.binomial(1,o)  

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
