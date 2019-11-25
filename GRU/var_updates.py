import numpy as np
from scipy.special import expit
from pypolyagamma import PyPolyaGamma
import sys

def update_normal_dim(prec, prec_muT):
    covar = np.linalg.solve(prec, np.identity(len(prec[:,0])))
    mu = covar @ prec_muT 
    return mu, covar

def update_bern(f):
    return expit(f)


def sample_post_pg(b,g,T,d):
    seed =np.random.randint(0,1000000000)
    pg = PyPolyaGamma(seed)
    n=T*d
    out = np.empty(n)
    g1 = g.ravel()
    pg.pgdrawv(b*np.ones(n), g1, out)
    return out.reshape(T,d,1)


def pg_update(bpg, h, u, W, U, b, T, d):
    g = W @ h[:-1,:,:] + U @ u + b
    return sample_post_pg(bpg,g,T,d)


def get_Wbar_params(z,omega,x,xxT, Sigma_inv,mu,T,d,ud):
    A = np.zeros((d, d+ud+1,d+ud+1))
    rhs = np.zeros((d,d+ud+1,1))
    for i in range(0,d):
        om_xxT = omega[:,i,:].reshape(T,1,1)*xxT

        A[i,:,:] = np.sum(om_xxT, axis = 0) + np.diag(Sigma_inv[i,:])

        zx = (z[:,i,:].reshape(T,1,1)-1/2)*x
        rhs[i,:,:] = np.sum(zx, axis=0)+(Sigma_inv[i,:]*mu[i,:]
                                         ).reshape(d+ud+1,1)
    W_covar = np.linalg.inv(A)
    W_mu = (W_covar @ rhs)#.reshape(d,d+ud+1)
    return W_covar, W_mu

def get_Wbar_y_params(x,xxT,y,Sigma_y_inv,Sigma_y_theta_inv,mu_prior,T,d,yd):
    A = np.zeros((yd,d+1,d+1))
    rhs = np.zeros((yd,d+1,1))
    Sig_y_inv = np.diagonal(Sigma_y_inv).reshape(yd,1)
    sum_xxT = np.sum(xxT,axis=0)
    for i in range(0,yd):        
        A[i,:,:]= np.diag(Sigma_y_theta_inv[i,:])+Sig_y_inv[i]*sum_xxT
        

        ydx = y[:,i,:].reshape(T,1,1)*x
    
        rhs[i,:,:] = Sig_y_inv[i]*np.sum(ydx, axis=0)

        rhs[i,:,:] += (Sigma_y_theta_inv[i,:]*mu_prior[i,:]).reshape(d+1,1)
    
    W_covar = np.linalg.inv(A)
    W_mu = (W_covar @ rhs)
    return W_covar, W_mu
    


def sample_weights(W_covar, W_mu, d, ud):
    #W_bar = np.zeros((d,d+ud+1))
    r = len(W_covar[:,0,0])
    W_bar = np.zeros((r,d+ud+1))
    #W_bar = np.zeros((d,d+ud+1))
    for j in range(0,r):
        W_bar[j,:] = np.random.multivariate_normal(W_mu[j,:,0], W_covar[j,:,:])
    return W_bar
        
def extract_W_weights(W_bar, d, ud):
    W = W_bar[:,:d]
    U = W_bar[:,d:d+ud]
    b = W_bar[:,-1].reshape(len(W_bar[:,-1]),1)
    return W, U, b

def Wbar_update(z,omega,x,xxT,Sigma_inv,mu,T,d,ud):
    W_covar, W_mu = get_Wbar_params(z,omega,x,xxT,Sigma_inv,mu,T,d,ud)
    W_bar = sample_weights(W_covar, W_mu, d, ud)
    W,U,b = extract_W_weights(W_bar, d, ud)
    return W_bar, W, U, b

def Wbar_y_update(x,xxT,y,Sigma_y_inv, Sigma_y_theta_inv, mu_prior,T,d,yd):
    W_covar, W_mu = get_Wbar_y_params(x,xxT,y,Sigma_y_inv,
                                      Sigma_y_theta_inv, mu_prior,T,d,yd)
    W_bar = sample_weights(W_covar, W_mu, d, 0)
    W,U,b = extract_W_weights(W_bar, d, 0)
    return W_bar, W, b


def init_weights(L,U, Sigma_theta, d, ud):
    r = len(Sigma_theta[:,0])
    W_mu_prior = np.random.uniform(L,U,size = (r,d+ud+1))
    W_bar = np.random.normal(W_mu_prior, Sigma_theta)
    W,U,b = extract_W_weights(W_bar, d, ud)
    return W_bar, W, U, b, W_mu_prior

    

