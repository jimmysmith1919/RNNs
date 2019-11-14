import numpy as np
import var_updates as update
import test6_build as build
import generate_GRU as gen
from scipy.special import expit
import  matplotlib.pyplot as plt
import sys

#seed = np.random.randint(0,100000)
#print(seed)
np.random.seed(60808)

T=3
d=3
ud = 2
h0 = .3*np.ones(d)
u = np.random.uniform(-1,1,size=(T,ud,1))
var=.01
inv_var = np.ones(d)*1/var


#Initialize weights
Sigma_theta = np.ones((d,d+ud+1))*(1/3)**2

L=-.9
U= .9
Wz_bar,Wz,Uz,bz,Wz_mu_prior = update.init_weights(L,U, Sigma_theta, d, ud)
Wr_bar,Wr,Ur,br,Wr_mu_prior  = update.init_weights(L,U, Sigma_theta, d, ud)
Wp_bar,Wp,Up,bp,Wp_mu_prior  = update.init_weights(L,U, Sigma_theta, d, ud)
'''
##TESTING###
Wz = np.random.uniform(3,4, size=(d,d))
Uz = np.random.uniform(3,4, size=(d,ud))
bz = np.random.uniform(3,4, size = (d,1))

Wr = np.random.uniform(3,4, size=(d,d))
Ur = np.random.uniform(3,4, size=(d,ud))
br = np.random.uniform(3,4, size = (d,1))

Wp = np.random.uniform(3,4, size=(d,d))
Up = np.random.uniform(3,4, size=(d,ud))
bp = np.random.uniform(3,4, size = (d,1))
'''

train_weights = True


#Initialize h
h = np.zeros((T+1,d,1))
h[0,:,0] = h0
Er = np.zeros((T,d,1))
Ez = np.zeros((T,d,1))
for i in range(1,T+1):
    #TESTING
    h[i,:,0] = np.random.multivariate_normal(h[i-1,:,0], np.diag(1/inv_var) )
    #np.random.uniform(-.5,.5,3 
    Er[i-1,:,:] = expit(Wr @ h[i-1,:,:] + Ur @ u[i-1,:,:] + br)
    Ez[i-1,:,:] = expit(Wz @ h[i-1,:,:] + Uz @ u[i-1,:,:] + bz)

#Initialize r 
r = np.random.binomial(1,Er, size=(T,d,1))
z = np.random.binomial(1,Ez, size=(T,d,1))

rh = np.zeros((T+1,d,1))
#Loop parameters

N=100000000
log_like_vec = []

h_samples =0
z_samples =0
r_samples =0
v_samples =0
Wz_bar_samples = 0
Wr_bar_samples = 0
Wp_bar_samples = 0

N_burn = .4*N

count = 0
for k in range(0,N):
    #Update pgs
    omega_z = update.pg_update(1, h, u, Wz, Uz, bz, T, d)
    omega_r = update.pg_update(1, h, u, Wr, Ur, br, T, d)
    
    rh[:-1,:,:] = r*h[:-1,:,:]
    gamma = update.pg_update(1, rh, u, 2*Wp, 2*Up, 2*bp, T, d)

    #Update v
    fv = build.build_v_param(h,z,rh,u,Wp,Up,bp, inv_var, d)
    Ev = update.update_bern(fv)
    v  = np.random.binomial(1, Ev, size=(T,d,1))
    
    #Update Zs
    fz = build.build_z_param(h,v,inv_var.reshape(d,1), Wz, Uz, u, bz, d)
    Ez = update.update_bern(fz)
    z = np.random.binomial(1,Ez, size=(T,d,1))

    

    #Update r's
    for j in range(0,d):                            
        frd = build.build_rd_param(h,u,v,gamma,r,Wp,Up,bp,Wr,Ur,br,j)
        Erd = update.update_bern(frd)
        
        r[:,j,:] = np.random.binomial(1,Erd, size=(T,1))

    
    #Update hs
    prec = build.build_prec_x(inv_var, Wz, omega_z, z, Wr, omega_r, r,
                              Wp, gamma, v, T, d)
    prec_muT = build.build_prec_muT(h0, u, inv_var, z, omega_z,
                                    Wz, Uz, bz,r, omega_r, Wr, Ur, br,
                                    v, Wp, Up, bp, gamma, T, d)
    mu, covar = update.update_normal_dim(prec,prec_muT)

    h = np.random.multivariate_normal(mu[:,0],covar)
    h = h.reshape(T,d,1)
    h = np.concatenate((h0.reshape(1,d,1), h), axis=0)

    if train_weights == True:
        #Update Weights
        x = np.concatenate((h[:-1,:,:],u, np.ones((T,1,1))), axis=1)
        xxT = (x[...,None]*x[:,None,:]).reshape(T,d+ud+1,d+ud+1)
        Wz_bar, Wz, Uz, bz = update.Wbar_update(z,omega_z,x,xxT,1/Sigma_theta,
                                           Wz_mu_prior,T,d,ud)
        Wr_bar, Wr, Ur, br = update.Wbar_update(r,omega_r,x,xxT,1/Sigma_theta,
                                           Wr_mu_prior,T,d,ud)
        #slight adjustment to x for tanh approx
        rx = 2*np.concatenate((h[:-1,:,:]*r,u, np.ones((T,1,1))), axis=1)
        rxrxT = (rx[...,None]*rx[:,None,:]).reshape(T,d+ud+1,d+ud+1)
        Wp_bar, Wp, Up, bp = update.Wbar_update(v, gamma, rx, rxrxT,
                                                1/Sigma_theta,
                                                Wp_mu_prior,T,d,ud)

        
    '''
    if k%100 == 0:
        loglike = build.log_like(h0,inv_var,h, z, omega, u, W, U, b, bpg,T, d)
        print(loglike)
        log_like_vec.append(loglike[0,0,0])
    '''
    if k > N_burn:
        h_samples +=h
        z_samples += z
        r_samples += r
        v_samples += v
        Wz_bar_samples += Wz_bar
        Wr_bar_samples += Wr_bar
        Wp_bar_samples += Wp_bar
    print(k)


Eh = h_samples/(N-N_burn-1)
Ez = z_samples/(N-N_burn-1)
Er = r_samples/(N-N_burn-1)
Ev = v_samples/(N-N_burn-1)
EWz_bar = Wz_bar_samples/(N-N_burn-1)
EWr_bar = Wr_bar_samples/(N-N_burn-1)
EWp_bar = Wp_bar_samples/(N-N_burn-1)
print('Eh')
print(Eh)
print('Ez')
print(Ez)
print('Er')
print(Er)
print('Ev')
print(Ev)
print('EWz_bar')
print(np.round(EWz_bar,4))
print('EWr_bar')
print(np.round(EWr_bar,4))
print('EWp_bar')
print(np.round(EWp_bar,4))



#Generate priors

M = 10000#N
h_samples = 0
z_samples = 0
r_samples = 0
v_samples = 0
Wz_bar_samples = 0
Wr_bar_samples = 0
Wp_bar_samples = 0


h1_samples = 0
z1_samples = 0
r1_samples = 0
v1_samples = 0


for n in range(0,M):
    if train_weights == True:
        Wz_bar = np.random.normal(Wz_mu_prior, Sigma_theta)
        Wz,Uz,bz = update.extract_W_weights(Wz_bar, d, ud)

        Wr_bar = np.random.normal(Wr_mu_prior, Sigma_theta)
        Wr,Ur,br = update.extract_W_weights(Wr_bar, d, ud)

        Wp_bar = np.random.normal(Wp_mu_prior, Sigma_theta)
        Wp,Up,bp = update.extract_W_weights(Wp_bar, d, ud)

    h = np.zeros((T+1,d,1))
    z = np.zeros((T,d,1))
    r = np.zeros((T,d,1))
    v = np.zeros((T,d,1))

    '''
    h1 = np.zeros((T+1,d,1))
    z1 = np.zeros((T,d,1))
    r1 = np.zeros((T,d,1))
    v1 = np.zeros((T,d,1))
    '''
    
    h[0,:,0] = h0

    #h1[0,:,0] = h0
    
    pz = expit(Wz @ h0.reshape(d) + Uz @ u[0,:,:].reshape(ud) + bz.reshape(d))
    z[0,:,0] =  np.random.binomial(1,pz)
    
    pr = expit(Wr @ h0.reshape(d) + Ur @ u[0,:,:].reshape(ud) + br.reshape(d))
    r[0,:,0] =  np.random.binomial(1,pr)


    pv = expit(2*(Wp @ (r[0,:,0]*h0).reshape(d)+
                  Up @ u[0,:,:].reshape(ud)+bp.reshape(d)))
    v[0,:,0] =  np.random.binomial(1,pv)
    
    Sigma = np.diag(1/inv_var)
    z_ch, r_ch, v_ch, h_ch, y = gen.stoch_GRU_step(Sigma,h[0,:,0],
                                                   u[0,:,0], Wz,
                                                   Uz,bz[:,0],
                                                   Wr, Ur, br[:,0],
                                                   Wp, Up, bp[:,0],
                                                   0, 0)
    
    
    for i in range(1,T+1):
        h_in = (1-z[i-1,:,0])*h[i-1,:,0]+z[i-1,:,0]*(2*v[i-1,:,0]-1)
        h[i,:,0] = np.random.multivariate_normal(h_in, 
                                                 np.diag(1/inv_var) )

        if i != T:
             pz = expit(Wz @ h[i,:,0].reshape(d) + 
                             Uz @ u[i,:,:].reshape(ud) + bz.reshape(d))
             z[i,:,0] = np.random.binomial(1,pz)

             pr = expit(Wr @ h[i,:,0].reshape(d) + 
                             Ur @ u[i,:,:].reshape(ud) + br.reshape(d))
             r[i,:,0] = np.random.binomial(1,pr)

             pv = expit(2*(Wp @ (r[i,:,0]*h[i,:,0]).reshape(d)+
                  Up @ u[i,:,:].reshape(ud)+bp.reshape(d)))
             v[i,:,0] =  np.random.binomial(1,pv)

             '''
             z_ch, r_ch, v_ch, h_ch, y = gen.stoch_GRU_step(Sigma,h[i,:,0],
                                                       u[i,:,0], Wz, Uz,
                                                       bz[:,0],
                                                       Wr, Ur, br[:,0],
                                                       Wp, Up, bp[:,0],
                                                       0, 0)
             '''
    
        
        
             
             
    h_samples += h
    z_samples += z
    r_samples += r
    v_samples += v
    Wz_bar_samples += Wz_bar
    Wr_bar_samples += Wr_bar
    Wp_bar_samples += Wp_bar

    '''
    h1_samples += h_ch 
    z1_samples += z_ch
    r1_samples += r_ch
    v1_samples += v_ch
    '''
    

Eh = h_samples/M
Ez = z_samples/M
Er = r_samples/M
Ev = v_samples/M
EWz_bar = Wz_bar_samples/M
EWr_bar = Wr_bar_samples/M
EWp_bar = Wp_bar_samples/M
print('Eh_prior')
print(Eh)
print('Ez_prior')
print(Ez)
print('Er_prior')
print(Er)
print('Ev_prior')
print(Ev)
print('EWz_bar prior')
print(np.round(EWz_bar,4))
print('EWr_bar prior')
print(np.round(EWr_bar,4))
print('EWp_bar prior')
print(np.round(EWp_bar,4))
#print('Wz_bar_prior_mean')
#print(np.round(Wz_mu_prior,4))


'''
Eh1 = h1_samples/M
Ez1 = z1_samples/M
Er1 = r1_samples/M
Ev1 = v1_samples/M

print('Eh1_prior')
print(Eh1)
print('Ez1_prior')
print(Ez1)
print('Er1_prior')
print(Er1)
print('Ev1_prior')
print(Ev1)
'''

'''
ran = np.arange(0,len(log_like_vec),1)
plt.plot(ran,log_like_vec)
plt.show()
'''

