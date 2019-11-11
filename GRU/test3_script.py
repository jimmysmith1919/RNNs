import numpy as np
import var_updates as update
import test3_build as build
from scipy.special import expit
import  matplotlib.pyplot as plt
import sys

#seed = np.random.randint(0,100000)
#print(seed)
np.random.seed(60808)

T=10
d=3
ud = 2
h0 = .3*np.ones(d)
u = np.random.uniform(-1,1,size=(T,ud,1))
var=.01
inv_var = np.ones(d)*1/var


#Initialize weights
Sigma_theta = np.ones((d,d+ud+1))*(1/3)**2
W_mu_prior = 10*np.random.uniform(-.1,.1,size = (d,d+ud+1))
W_bar = np.random.normal(W_mu_prior, Sigma_theta)
W,U,b = update.extract_W_weights(W_bar, d, ud)

train_weights = True

'''
W = np.random.uniform(-1,1, size=(d,d))
U = np.random.uniform(-1,1, size=(d,ud))
b = np.random.uniform(-1,1, size = (d,1))
'''

#Initialize h
h = np.zeros((T+1,d,1))
h[0,:,0] = h0
for i in range(1,T+1):
    h[i,:,0] = np.random.multivariate_normal(h[i-1,:,0], np.diag(1/inv_var) )
    


#Loop parameters

N=100000
log_like_vec = []

h_samples = 0
z_samples =0
W_bar_samples = 0

N_burn = .4*N

count = 0
for k in range(0,N):
    #Update omegas
    g = W @ h[:-1,:,:] + U @ u + b
    bpg=1
    omega = update.sample_post_pg(bpg,g,T,d)
    
    #Update Zs
    fz = build.build_z_param(h, inv_var.reshape(d,1), W, U, u, b, d)
    Ez = update.update_bern(fz)
    z = np.random.binomial(1,Ez, size=(T,d,1))

    #Update hs
    prec = build.build_prec_x(inv_var, W, omega, z, T,d)
    prec_muT = build.build_prec_muT(h0, u, inv_var, z, omega, W, U, b, T, d)
    mu, covar = update.update_normal_dim(prec,prec_muT)

    h = np.random.multivariate_normal(mu[:,0],covar)
    h = h.reshape(T,d,1)
    h = np.concatenate((h0.reshape(1,d,1), h), axis=0)

    if train_weights == True:
        #Update Weights
        x = np.concatenate((h[1:,:,:],u, np.ones((T,1,1))), axis=1)
        xxT = (x[...,None]*x[:,None,:]).reshape(T,d+ud+1,d+ud+1)

    
        W_covar, W_mu = update.Wbar_update(z,omega,x,xxT,1/Sigma_theta,
                                           W_mu_prior,T,d,ud)
        W_bar = update.sample_weights(W_covar, W_mu, d, ud)
        W,U,b = update.extract_W_weights(W_bar, d, ud)
    
    
    '''
    if k%100 == 0:
        loglike = build.log_like(h0,inv_var,h, z, omega, u, W, U, b, bpg,T, d)
        print(loglike)
        log_like_vec.append(loglike[0,0,0])
    '''
    if k > N_burn:
        h_samples +=h
        z_samples += z
        W_bar_samples += W_bar
    print(k)


Eh = h_samples/(N-N_burn-1)
Ez = z_samples/(N-N_burn-1)
EW_bar = W_bar_samples/(N-N_burn-1)
print('Eh')
print(Eh)
print('Ez')
print(Ez)
print('EW_bar')
print(np.round(EW_bar,4))



#Generate priors
z_prior = expit(W @ h0.reshape(1,d,1) + U @ u[0,:,:] + b)

M = N
h_samples = 0
z_samples = 0 
W_bar_samples = 0



for n in range(0,M):
    if train_weights == True:
        W_bar = np.random.normal(W_mu_prior, Sigma_theta)
        W,U,b = update.extract_W_weights(W_bar, d, ud)

    h = np.zeros((T+1,d,1))
    z = np.zeros((T,d,1))
    h[0,:,0] = h0
    pz = expit(W @ h0.reshape(d) + U @ u[0,:,:].reshape(ud) + b.reshape(d))
    z[0,:,0] =  np.random.binomial(1,pz)
    for i in range(1,T+1):
        h[i,:,0] = np.random.multivariate_normal((1-z[i-1,:,0])*h[i-1,:,0], 
                                                 np.diag(1/inv_var) )
        if i != T:
             pz = expit(W @ h[i,:,0].reshape(d) + 
                             U @ u[i,:,:].reshape(ud) + b.reshape(d))
             z[i,:,0] = np.random.binomial(1,pz)
    h_samples += h
    z_samples += z
    W_bar_samples += W_bar



Eh = h_samples/M
Ez = z_samples/M
EW_bar = W_bar_samples/M
print('Eh_prior')
print(Eh)
print('Ez_prior')
print(Ez)
print('EW_bar prior')
print(np.round(EW_bar,4))
print('W_bar_prior_mean')
print(np.round(W_mu_prior,4))

r = np.arange(0,len(log_like_vec),1)
plt.plot(r,log_like_vec)
plt.show()
