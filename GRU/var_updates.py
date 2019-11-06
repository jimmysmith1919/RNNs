import numpy as np
from scipy.special import expit
from pypolyagamma import PyPolyaGamma


def update_normal_dim(prec, prec_muT):
    covar = np.linalg.inv(prec)
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
    
 
def Wbar_update(z,omega,x,xxT,T,d,ud):
    A = np.zeros((d, d+ud+1,d+ud+1))
    rhs = np.zeros((d,d+ud+1,1))
    for i in range(0,d):
        om_xxT = omega[:,i,:].reshape(T,1,1)*xxT

        A[i,:,:] = np.sum(om_xxT, axis = 0)

        zx = (z[:,i,:].reshape(T,1,1)-1/2)*x
        rhs[i,:,:] = np.sum(zx, axis=0)
    return np.linalg.solve(A,rhs).reshape(d,d+ud+1)


