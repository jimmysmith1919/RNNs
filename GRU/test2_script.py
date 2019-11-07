import numpy as np
import var_updates as update
import test2_build as build
from scipy.special import expit
import  matplotlib.pyplot as plt
import sys

#seed = np.random.randint(0,100000)
#print(seed)
np.random.seed(60808)

T=10
d=3
ud = 2
h0 = -2*np.ones(d)
u = np.random.uniform(-1,1,size=(T,ud,1))
var=.01
inv_var = np.ones(d)*1/var


#Initialize weights
W = np.random.uniform(-1,1, size=(d,d))
U = np.random.uniform(-1,1, size=(d,ud))
b = np.random.uniform(-1,1, size = (d,1))





#Initialize h
h = np.zeros((T+1,d,1))
h[0,:,0] = h0
for i in range(1,T+1):
    h[i,:,0] = np.random.multivariate_normal(h[i-1,:,0], np.diag(1/inv_var) )
    


#Loop parameters

N=1000
log_like_vec = []

h_samples = 0
z_samples =0

N_burn = .4*N

count = 0
for k in range(0,N):
    g = W @ h[:-1,:,:] + U @ u + b
    bpg=1
    omega = update.sample_post_pg(bpg,g,T,d)
    

    fz = W @ h[:-1,:,:] + U @ u + b
    Ez = update.update_bern(fz)
    z = np.random.binomial(1,Ez, size=(T,d,1))

    

    prec = build.build_prec_x(inv_var,W, omega, T,d)
    prec_muT = build.build_prec_muT(h0, u, inv_var, z, omega, W, U, b, T, d)

    mu, covar = update.update_normal_dim(prec,prec_muT)

    h = np.random.multivariate_normal(mu[:,0],covar)
    h = h.reshape(T,d,1)
    h = np.concatenate((h0.reshape(1,d,1), h), axis=0)

    '''
    x = np.concatenate((h,u, np.ones((1,1,1))), axis=1)
    xxT = (x[...,None]*x[:,None,:]).reshape(T,d+ud+1,d+ud+1)
    print(np.linalg.matrix_rank(xxT))
    Wbar = update.Wbar_update(z,omega,x,xxT,T,d,ud)
    print(Wbar)
    '''
    '''
    if k%100 == 0:
        loglike = build.log_like(h0,inv_var,h, z, omega, u, W, U, b, bpg,T, d)
        print(loglike)
        log_like_vec.append(loglike[0,0,0])
    '''
    if k > N_burn:
        h_samples +=h
        z_samples += z
        
    print(k)


Eh = h_samples/(N-N_burn-1)
Ez = z_samples/(N-N_burn-1)
print('Eh')
print(Eh)
print('Ez')
print(Ez)



#Generate priors
z_prior = expit(W @ h0.reshape(1,d,1) + U @ u[0,:,:] + b)

M = N
h_samples = 0
z_samples = 0 

for n in range(0,M):
    h = np.zeros((T+1,d,1))
    z = np.zeros((T,d,1))
    h[0,:,0] = h0
    z[0,:,0] = expit(W @ h0.reshape(d) + U @ u[0,:,:].reshape(ud) + b.reshape(d))
    for i in range(1,T+1):
        h[i,:,0] = np.random.multivariate_normal(h[i-1,:,0], np.diag(1/inv_var) )
        if i != T:
            z[i,:,0] = expit(W @ h[i,:,0].reshape(d) + U @ u[i,:,:].reshape(ud) + b.reshape(d)) 
    
    h_samples += h
    z_samples += z



h = h_samples/M
z = z_samples/M
print('h_prior')
print(h)
print('z_prior')
print(z)


r = np.arange(0,len(log_like_vec),1)
plt.plot(r,log_like_vec)
plt.show()
