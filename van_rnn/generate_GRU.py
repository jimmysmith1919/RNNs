import numpy as np
from scipy.special import expit
import sys
import var_updates as update
from scipy.stats import norm

def RNN_step(h_0, u, Wp, Up, bp, Wy, by):
    fp = Wp @ h_0 + Up @ u + bp
    h = np.tanh(fp)
    
    y = Wy @ h + by
    return h, y


def stoch_RNN_step_mean(h_0, u, 
                        Wp, Up, bp, Wy, by, alpha, tau):
    d = len(h_0)
    
    fp = 2*(Wp @ h_0 + Up @ u + bp)
    
    zeta1 = (-fp-alpha)/tau
    zeta2 = (fp-alpha)/tau

    
    pv = np.zeros((d,3))
    pv[:,0] = expit(zeta1)
    pv[:,1] = np.exp(np.log(expit(zeta2))+np.log(expit(-zeta1)))
    pv[:,2] = 1-(pv[:,0]+pv[:,1])
    
    
    V1 = np.zeros(d)
    V2 = np.ones(d)
    V3 = 1/(2*alpha)*fp+1/2

    EV = V1*pv[:,0]+V2*pv[:,1]+V3*pv[:,2]


    Eh = 2*EV-1
    Ey = Wy @ Eh + by
    return EV, Eh, Ey





def stoch_RNN_step(Sigma,h_0, u, Wp, Up, bp,
                   Sigma_y, Wy, by, alpha, tau):
    d = len(h_0)

    fp = 2*(Wp @ h_0 + Up @ u + bp)

    zeta1 = (-fp-alpha)/tau
    zeta2 = (fp-alpha)/tau

    
    pv = np.zeros((d,3))
    pv[:,0] = expit(zeta1)
    pv[:,1] = np.exp(np.log(expit(zeta2))+np.log(expit(-zeta1)))
    pv[:,2] = 1-(pv[:,0]+pv[:,1])
    
    v = update.vectorized_cat(pv, np.arange(3))
    v = np.eye(3)[v]
    v= v.reshape(1,d,3)

    V = big_V(1, v, alpha, tau, fp, d)
    V = V.reshape(d)
    Eh = 2*V-1
    h = np.random.multivariate_normal(Eh, Sigma)
    
    Ey = Wy @ h + by
    y= np.random.multivariate_normal(Ey, Sigma_y)
    #y = norm.rvs(loc=Ey, scale = np.sqrt(np.diag(Sigma_y)))
    return  v, h, y

def big_V(M,v, alpha, tau, fp,d):
    V1 = np.zeros((M,d,1))
    V2 = np.ones((M,d,1))
    V3 = (1/(2*alpha)*fp+1/2).reshape((M,d,1))
    
    V = np.zeros((M,d))
    V += v[:,:,0]*V1[:,:,0]
    V += v[:,:,1]*V2[:,:,0]
    V += v[:,:,2]*V3[:,:,0]
    #V = V.reshape(d)
    return V
    


def stoch_RNN_step_vectorized(M,d,Sigma,h_0, u, Wp_bar,
                   Sigma_y, Wy, by, alpha, tau):


    newu = u*np.ones((M,len(u),1))
    ones = np.ones((M,1,1))
    
    x = np.concatenate((h_0, newu, ones), axis=1)
    
    
    
    fp = 2*(Wp_bar @ x)
    
    

    
    zeta1 = (-fp-alpha)/tau
    zeta2 = (fp-alpha)/tau
    
    pv = np.zeros((M,d,3))
    pv[:,:,0] += expit(zeta1)[:,:,0]
    pv[:,:,1] += np.exp(np.log(expit(zeta2))+np.log(expit(-zeta1)))[:,:,0]

    pv[:,:,2] += 1-(pv[:,:,0]+pv[:,:,1])

    
    v = update.vectorized_vector_cat(pv, np.arange(3), M)
    v = np.eye(3)[v]    

    '''
    V1 = np.zeros((M,d,1))
    V2 = np.ones((M,d,1))
    V3 = 1/(2*alpha)*fp+1/2
    
    V = np.zeros((M,d))
    V += v[:,:,0]*V1[:,:,0]
    V += v[:,:,1]*V2[:,:,0]
    V += v[:,:,2]*V3[:,:,0]
    '''
    V = big_V(M, v, alpha, tau, fp, d)
    
    Eh = (2*V-1)
    scale = np.sqrt(Sigma)*np.ones(Eh.shape)
    Eh = Eh.reshape(M,d,1)
    scale = scale.reshape(M,d,1)
    h = ( norm.rvs(loc=Eh, scale=scale) ).reshape(M,d,1)

    Ey = Wy @ h + by

    yd = Ey.shape[1]
    scale = np.sqrt(np.diag(Sigma_y)).reshape(1,yd,1)*np.ones(Ey.shape)
    y= norm.rvs(loc=Ey, scale=scale)
        
    return  V, v, h, y





def generate_rec_inp(func, steps, h_0, u, 
             Wi, Ui, bi, 
             Wr, Ur, br, 
             Wp, Up, bp, 
             Wy, by):
    '''For recurrent inputs where previous y is next input u '''
    
    d = len(h_0)
    yd = len(u)
    i_vec = np.zeros((steps, d))
    r_vec = np.zeros((steps, d))
    p_vec = np.zeros((steps, d))
    h_vec = np.zeros((steps, d))
    y_vec = np.zeros((steps, yd))
    
    h = h_0
    for step in range(0,steps):
        i,r,p,h,y = func(h, u, 
                         Wi, Ui, bi, 
                         Wr, Ur, br, 
                         Wp, Up, bp, 
                         Wy, by)
        i_vec[step,:] = i
        r_vec[step,:] = r
        p_vec[step,:] = p
        h_vec[step,:] = h
        y_vec[step,:] = y
        
        u = y
    return i_vec, r_vec, p_vec, h_vec, y_vec
