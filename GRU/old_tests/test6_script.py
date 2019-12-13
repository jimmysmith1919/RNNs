import numpy as np
import var_updates as update
import test6_build as build
import generate_GRU as gen
import plot_dist as plot_dist
from scipy.special import expit
import  matplotlib.pyplot as plt
import sys

#seed = np.random.randint(0,100000)
#print(seed)
np.random.seed(60808)

T=4
d=3
ud = 2
h0 = 0*np.ones(d)
u = np.random.uniform(-1,1,size=(T,ud,1))
var=.1
inv_var = np.ones(d)*1/var


#Initialize weights
Sigma_theta = np.ones((d,d+ud+1))*(1/3)**2

L=-.9
U= .9
Wz_bar,Wz,Uz,bz,Wz_mu_prior = update.init_weights(L,U, Sigma_theta, d, ud)
Wr_bar,Wr,Ur,br,Wr_mu_prior  = update.init_weights(L,U, Sigma_theta, d, ud)
Wp_bar,Wp,Up,bp,Wp_mu_prior  = update.init_weights(L,U, Sigma_theta, d, ud)

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

N=10000
log_like_vec = []

h_samples =0
z_samples =0
r_samples =0
v_samples =0
Wz_bar_samples = 0
Wr_bar_samples = 0
Wp_bar_samples = 0

N_burn = int(.4*N)

T_check = 4
d_check = 2
h_samples_vec = np.zeros((N-N_burn-1,d))



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

        
    if k > N_burn:
        h_samples_vec[k-N_burn-1,:] = h[T_check,:,0]
        h_samples += h
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


'''
#Generate priors

M = 100000#N
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

    
    h[0,:,0] = h0

    
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

    
             
    h_samples += h
    z_samples += z
    r_samples += r
    v_samples += v
    Wz_bar_samples += Wz_bar
    Wr_bar_samples += Wr_bar
    Wp_bar_samples += Wp_bar

    
    

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

'''
if train_weights == True:
    Wz, Uz, bz = update.extract_W_weights(Wz_mu_prior, d, ud)
    Wr, Ur, br = update.extract_W_weights(Wr_mu_prior, d, ud)
    Wp, Up, bp = update.extract_W_weights(Wp_mu_prior, d, ud)

print('h0')    
print(h0)
if T_check>1:
    Ezi, Ezr, Ev, Eh, Ey = gen.stoch_GRU_step_mean(h0, u[0,:,0],
                                                   Wz, Uz, bz.reshape(d),
                                                   Wr, Ur, br.reshape(d),
                                                   Wp, Up, bp.reshape(d), 0, 0)
    for t in range(1,T_check-1):
        Ezi, Ezr, Ev, Eh, Ey = gen.stoch_GRU_step_mean(Eh, u[t,:,0],
                                                   Wz, Uz, bz.reshape(d),
                                                   Wr, Ur, br.reshape(d),
                                                   Wp, Up, bp.reshape(d), 0, 0)
        print(t)
    h0 = Eh

print('h0')
print(h0)
'''
'''        
bin_vec = plot_dist.get_binary()
x = np.linspace(-2,2,1000)


mix_pdf, mix_mean, pv_mean, pz, pr = plot_dist.get_likelihood(x,
                    bin_vec, h0, var,
                     Wz, Uz, bz.reshape(d),
                     Wr, Ur, br.reshape(d),
                     Wp, Up, bp.reshape(d), u[T_check-1,:,0])
'''

'''
print('mix_mean')
print(mix_mean)
print('pv_mean')
print(pv_mean)
print('pz')
print(pz)
print('pr')
print(pr)
'''

'''
plt.plot(x, mix_pdf, 'r', label='pdf')
plt.hist(h_samples_vec[:,0].reshape(N-N_burn-1), bins=100, density=True)
plt.xlabel('h_t')
plt.ylabel('P(h_t|h_{t-1})')
plt.title('Gibbs: T={}, T_check={}, N={}, Var={}'.format(
    T,T_check,N,var))
plt.legend()
plt.show()
'''


M=10000

z_vec = np.zeros((M,d))
r_vec = np.zeros((M,d))
v_vec = np.zeros((M,d))
h_vec = np.zeros((M,d))

if train_weights == True:
    Wz, Uz, bz = update.extract_W_weights(Wz_mu_prior, d, ud)
    Wr, Ur, br = update.extract_W_weights(Wr_mu_prior, d, ud)
    Wp, Up, bp = update.extract_W_weights(Wp_mu_prior, d, ud)


for i in range(0,M):
    h = h0
    for t in range(0,T_check):
        z, r, v, h, y = gen.stoch_GRU_step(np.diag(1/inv_var), h, u[t,:,0],
                                       Wz, Uz, bz.reshape(d),
                                       Wr, Ur, br.reshape(d),
                                       Wp, Up, bp.reshape(d), 0, 0)

    z_vec[i,:] = z
    r_vec[i,:] = r
    v_vec[i,:] = v
    h_vec[i,:] = h



#plt.plot(x, mix_pdf, 'r', label='pdf')
plt.hist(h_samples_vec[:,d_check].reshape(N-N_burn-1), bins=100, density=True)
plt.hist(h_vec[:,d_check], bins=100, histtype='step',color='r', density=True, label='prior')
plt.xlabel('h_t')
plt.ylabel('P(h_t|h_{t-1})')
plt.title('Gen: T={}, T_check={}, d_check={}, N={}, Var={}'.format(
    T,T_check,d_check,N,var))
plt.legend()
plt.show()
    

'''
plt.plot(x, mix_pdf, 'r', label='pdf')
plt.hist(h_vec, bins=100, density=True)
plt.xlabel('h_t')
plt.ylabel('P(h_t|h_{t-1})')
plt.title('Samples from  Generative function')
plt.legend()
plt.show()
'''
