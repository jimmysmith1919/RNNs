import numpy as np
import var_updates as update
import test1_build as build
from scipy.special import expit
import  matplotlib.pyplot as plt
import sys

#seed = np.random.randint(0,100000)
#print(seed)
np.random.seed(60808)

T=1
d=3
ud = 2
h0 = -2*np.ones(d)
u = np.random.uniform(-1,1,size=(T,ud,1))
var=.3
inv_var = np.ones(d)*1/var


#Initialize weights
W = np.random.uniform(-1,1, size=(d,d))
U = np.random.uniform(-1,1, size=(d,ud))
b = np.random.uniform(-1,1, size = (d,1))


z_prior = expit(W @ h0.reshape(T,d,1) + U @ u + b)

#Initialize h
h = np.random.multivariate_normal(h0, np.diag(1/inv_var) )
h = h.reshape(T,d,1)




#Loop parameters
diff = np.inf
N=10000

log_like_vec = []

h_samples = 0
z_samples =0

N_burn = .4*N


for k in range(0,N):
    g = W @ h + U @ u + b
    bpg=1
    omega = update.sample_post_pg(bpg,g,T,d)

    fz = W @ h + U @ u + b
    Ez = update.update_bern(fz)
    z = np.random.binomial(1,Ez, size=(T,d,1))


    prec = build.build_prec_x(inv_var,W, omega[0,:,0])
    prec_muT = build.build_prec_muT(h0, u, inv_var, z, omega, W, U, b, T, d)
    mu, covar = update.update_normal_dim(prec,prec_muT)
    h = np.random.multivariate_normal(mu[0,:,0],covar)
    h = h.reshape(T,d,1)
  
    '''
    x = np.concatenate((h,u, np.ones((1,1,1))), axis=1)
    xxT = (x[...,None]*x[:,None,:]).reshape(T,d+ud+1,d+ud+1)
    print(np.linalg.matrix_rank(xxT))
    Wbar = update.Wbar_update(z,omega,x,xxT,T,d,ud)
    print(Wbar)
    '''

    if k%100 == 0:
        loglike = build.log_like(h0,inv_var,h, z, omega, u, W, U, b, bpg,T, d)
        print(loglike)
        log_like_vec.append(loglike[0,0,0])
    
    if k > N_burn:
        h_samples +=h
        z_samples += z
        
    print(k)

Eh = h_samples/(N-N_burn)
Ez = z_samples/(N-N_burn)
print(Eh)
print(Ez)
print('z_prior')
print(z_prior)

r = np.arange(0,len(log_like_vec),1)
plt.plot(r,log_like_vec)
plt.show()
