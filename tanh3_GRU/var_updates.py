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

def update_cat3(cond):
    print(cond.shape)
    print(cond)
    return np.random.multinomial(1,cond)

def vectorized_cat(prob_matrix, items):
    prob_matrix=prob_matrix.T
    s = prob_matrix.cumsum(axis=0)
    r = np.random.rand(prob_matrix.shape[1])
    k = (s < r).sum(axis=0)
    return items[k]



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
    return sample_post_pg(bpg,g,T,d), g

def sample_post_gamma(b,g,T,d):
    seed =np.random.randint(0,1000000000)
    pg = PyPolyaGamma(seed)
    n=T*d*2
    out = np.empty(n)
    pg.pgdrawv(b, g, out)

    half = int(len(out)/2)
    half1 = out[:half].reshape(T,d,1)
    half2 =out[half:].reshape(T,d,1)

    out = np.concatenate((half1, half2), axis= 2)
    return out


def gamma_update(h, v, u, W, U, b, T, d, alpha, tau):
    #Plug in rh for h
    fp = W @ h[:-1,:,:] + U @ u + b
    zeta1 = (-fp-alpha)/tau
    zeta2 = (fp-alpha)/tau
    zeta = np.concatenate((zeta1,zeta2), axis=2)

    v_cat = np.argmax(v,axis=2)
    v_cat_flat = v_cat.ravel()
    
    indgreat1 = v_cat_flat >= 0
    indgreat2 = v_cat_flat >= 1
    indgreat = np.concatenate((indgreat1, indgreat2))
    
    zeta_flat = np.concatenate((zeta1.ravel(), zeta2.ravel()))
    return sample_post_gamma(
        indgreat.astype(np.double),zeta_flat,T,d ), zeta_flat, indgreat



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
    W_mu = (W_covar @ rhs)
    return W_covar, W_mu


# get_Wbar_params(z,omega,x,xxT, Sigma_inv,mu,T,d,ud):

def get_Wbar_p_params(h, z,gamma,v,x,xxT,Sigma_inv,mu,T,d, ud,
                      alpha, tau, inv_var):
    h = h[1:]
    A = np.zeros((d, d+ud+1,d+ud+1))
    rhs = np.zeros((d,d+ud+1,1))
    for i in range(0,d):
        mid_term = (1/tau**2)*(gamma[:,i,0].reshape(T,1,1) +
                               gamma[:,i,1].reshape(T,1,1) )*xxT
        
        check = v[:,i,2]==1

        last_jh_term = 0
        
        if np.sum(check) > 0:
            z_check=z[check,i].reshape(-1,1,1)
            zrx_term = np.zeros(mid_term.shape)
            zrx_term[check] += inv_var[i]*(1/alpha**2)*z_check*xxT[check]
            mid_term += zrx_term


            last_jh_term = np.zeros((x.shape))
            
            last_jh_term[check] += h[check,i].reshape(-1,1,1)*inv_var[i]*(
                1/alpha)*z_check*x[check]
             
        
        A[i,:,:] = np.sum(mid_term, axis = 0) + np.diag(Sigma_inv[i,:])
        

        
        ind1 = v[:,i,0]==1
        ind1 = ind1.reshape(-1,1,1)

        ind2 = v[:,i,1]==1
        ind2 = ind2.reshape(-1,1,1)

        ind_great_2 = v[:,i,0]==0
        ind_great_2 = ind_great_2.reshape(-1,1,1)

        jh = -1/tau*(ind1-1/2)*x
        jh += 1/tau*(ind2-ind_great_2/2)*x
        
        jh += alpha*(1/tau**2)*(-gamma[:,i,0].reshape(T,1,1) +
                              gamma[:,i,1].reshape(T,1,1) )*x
        jh += last_jh_term


        rhs[i,:,:] = np.sum(jh, axis=0)+(Sigma_inv[i,:]*mu[i,:]
                                         ).reshape(d+ud+1,1)
    W_covar = np.linalg.inv(A)
    W_mu = (W_covar @ rhs)
    
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
    
def test_Wbar_y(y,x,W_bar_y, Sigma_y_inv, Sigma_y_theta_inv, mu_prior, T):
    sum_p = 0
    for t in range(0,T):
        sum_p += -1/2*( (y[t,0,0]-W_bar_y @ x[t,0,0]).T @
                        Sigma_y_inv @ (y[t,0,0]-W_bar_y @ x[t,0,0]) )

    return sum_p


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
    return W_bar, W, U, b, W_mu, W_covar



def Wbar_p_update(h, z,gamma,v,x,xxT,Sigma_inv,mu,T,d,ud, alpha, tau, inv_var):
    W_covar, W_mu = get_Wbar_p_params(h, z,gamma,v,x,xxT,
                                      Sigma_inv,mu,T,d, ud,
                                      alpha, tau, inv_var)
    W_bar = sample_weights(W_covar, W_mu, d, ud)
    W,U,b = extract_W_weights(W_bar, d, ud)
    return W_bar, W, U, b, W_mu, W_covar

def Wbar_y_update(x,xxT,y,Sigma_y_inv, Sigma_y_theta_inv, mu_prior,T,d,yd):
    W_covar, W_mu = get_Wbar_y_params(x,xxT,y,Sigma_y_inv,
                                      Sigma_y_theta_inv, mu_prior,T,d,yd)
    W_bar = sample_weights(W_covar, W_mu, d, 0)
    W,U,b = extract_W_weights(W_bar, d, 0)
    return W_bar, W, b, W_mu, W_covar


def init_weights(L,U, Sigma_theta, d, ud):
    r = len(Sigma_theta[:,0])
    W_mu_prior = np.random.uniform(L,U,size = (r,d+ud+1))
    W_bar = np.random.normal(W_mu_prior, np.sqrt(Sigma_theta))
    W,U,b = extract_W_weights(W_bar, d, ud)
    return W_bar, W, U, b, W_mu_prior

    

