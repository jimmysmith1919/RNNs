import numpy as np
import var_updates as update
import build_model as build
import main_loop as loop
import generate_GRU as gen
import plot_dist as plot_dist
from scipy.special import expit
import  matplotlib.pyplot as plt
import sys


seed = 4571#np.random.randint(0,10000)
np.random.seed(seed)
print('random_seed:',seed)


## DATA  #######
#Sine Wave#
end = 800
dt = 1
T_new = 100


t = np.arange(0, end, dt)
#data = np.sin((.06+.006)*t)
data = np.sin((.2)*t)
ud = 1
yd = 1

stop = int(.8*end/dt)
T = stop-1
u = data[:T].reshape(T,ud,1)
y = data[1:stop].reshape(T,yd,1)
y_last = y[-1,:,0]

T_full = len(data)-1
y_full = data[1:].reshape(T_full, ud, 1)

T_test = T_full-T
#u_test = data[stop-1:-1].reshape(T_test,ud,1)
y_test = data[stop:].reshape(T_test,yd,1)
#####################


##Model Initialization
d=10
h0 = 0*np.ones(d)
var=.3
inv_var = np.ones(d)*1/var
var_y = .1
Sigma_y = var_y*np.identity(yd)
Sigma_y_inv = 1/var_y*np.identity(yd)

#Initialize weights
theta_var = (1/3)**2
Sigma_theta = np.ones((d,d+ud+1))*theta_var

theta_y_var = (1/3**2)
Sigma_y_theta = np.ones((yd,d+1))*theta_y_var

L=-.9
U= .9

'''
Wz_bar,Wz,Uz,bz,Wz_mu_prior = update.init_weights(L,U, Sigma_theta, d, ud)
Wr_bar,Wr,Ur,br,Wr_mu_prior  = update.init_weights(L,U, Sigma_theta, d, ud)
Wp_bar,Wp,Up,bp,Wp_mu_prior  = update.init_weights(L,U, Sigma_theta, d, ud)

Wy_bar,Wy,_,by,Wy_mu_prior  = update.init_weights(L,U, Sigma_y_theta, d, 0)
'''

train_weights = False

##Load Weights
wfile = 'weights/'+'GRU_d10_eps150_lr0.0001_end100_1576695162.8768609.npy'
weights = np.load(wfile, allow_pickle=True)

Uz = (weights[0][0,:d]).reshape(d,ud)
Ur = (weights[0][0,d:2*d]).reshape(d,ud)
Up = (weights[0][0,2*d:3*d]).reshape(d,ud)


Wz = weights[1][:,:d].T
Wr = weights[1][:,d:2*d].T
Wp = weights[1][:,2*d:3*d].T


bz = (weights[2][:d]).reshape(d,1)
br = (weights[2][d:2*d]).reshape(d,1)
bp = (weights[2][2*d:3*d]).reshape(d,1)

Wy = weights[3].reshape(1,d)
by = weights[4].reshape(1,1)



Wz_mu_prior = np.concatenate((Wz,Uz,bz), axis = 1)
Wr_mu_prior = np.concatenate((Wr,Ur,br), axis = 1) 
Wp_mu_prior = np.concatenate((Wp,Up,bp), axis = 1) 
Wy_mu_prior = np.concatenate((Wy,by), axis = 1) 



#Initialize h
h = np.zeros((T+1,d,1))
h[0,:,0] = h0
r = np.zeros((T,d,1))
z = np.zeros((T,d,1))
for j in range(0,T):
    zt, rt, _, ht, _ = gen.stoch_GRU_step(np.diag(1/inv_var), h0, u[j,:,0],
                                       Wz, Uz, bz.reshape(d),
                                       Wr, Ur, br.reshape(d),
                                       Wp, Up, bp.reshape(d),
                                       Sigma_y, Wy, by.reshape(yd))
    r[j,:,0] = rt
    z[j,:,0] = zt
    h[j+1,:,0] = ht

    
rh = np.zeros((T+1,d,1))


#Loop parameters
N=10000
M=1000  #number of test samples
N_burn = int(.4*N)
T_check = -1 
d_check = 0

h_samples, z_samples, r_samples, v_samples, Wz_bar_samples,Wr_bar_samples, Wp_bar_samples, Wy_bar_samples, h_samples_vec, Wz_bar_samples_vec, Wr_bar_samples_vec, Wp_bar_samples_vec, Wy_bar_samples_vec = loop.gibbs_loop(N, N_burn, T, d,
                                                 T_check, ud, yd, h0,
                                                 inv_var, Sigma_y_inv,
                                                 Sigma_theta, Sigma_y_theta,
                                                 Sigma_y,
                                                 Wz_mu_prior, Wr_mu_prior,
                                                 Wp_mu_prior, Wy_mu_prior, Wz,
                                                 Uz, bz, Wr, Ur, br, Wp, Up,
                                                 bp, Wy, by, train_weights,
                                                 u, y, h, r, rh, z)


Eh = h_samples/(N-N_burn-1)
Ez = z_samples/(N-N_burn-1)
Er = r_samples/(N-N_burn-1)
Ev = v_samples/(N-N_burn-1)
EWz_bar = Wz_bar_samples/(N-N_burn-1)
EWr_bar = Wr_bar_samples/(N-N_burn-1)
EWp_bar = Wp_bar_samples/(N-N_burn-1)
EWy_bar = Wy_bar_samples/(N-N_burn-1)

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
print('EWy_bar')
print(np.round(EWy_bar,4))



#Training Predictions

#Mean weights
Wz,Uz,bz = update.extract_W_weights(EWz_bar, d, ud)
Wr,Ur,br = update.extract_W_weights(EWr_bar, d, ud)
Wp,Up,bp = update.extract_W_weights(EWp_bar, d, ud)
Wy,_,by = update.extract_W_weights(EWy_bar, d, 0)


h = h0
train_y = np.zeros((T, yd))

for j in range(0, T):
    z, r, v, h, yt = gen.stoch_GRU_step_mean(h, u[j,:,0],
                                             Wz, Uz, bz.reshape(d),
                                             Wr, Ur, br.reshape(d),
                                             Wp, Up, bp.reshape(d),
                                             Wy, by.reshape(yd))
    train_y[j,:] = yt

plt.plot(t[1:],y_full.reshape(T_full), label = 'True')
plt.plot(t[1:stop], train_y.reshape(T), label = 'mean_train')



samples_len = len(h_samples_vec[:,0,0])

train_y_vec = np.zeros((M,T))


for i in range(0,M):
    h = h0
    
    Wz_bar = Wz_bar_samples_vec[samples_len-M+i]
    Wz,Uz,bz = update.extract_W_weights(Wz_bar, d, ud)

    Wr_bar = Wr_bar_samples_vec[samples_len-M+i]
    Wr,Ur,br = update.extract_W_weights(Wr_bar, d, ud)

    Wp_bar = Wp_bar_samples_vec[samples_len-M+i]
    Wp,Up,bp = update.extract_W_weights(Wp_bar, d, ud)

    Wy_bar = Wy_bar_samples_vec[samples_len-M+i]
    Wy,_,by = update.extract_W_weights(Wy_bar, d, 0)
    
    for j in range(0,T):
        
        z, r, v, h, y = gen.stoch_GRU_step(np.diag(1/inv_var), h, u[j,:,0],
                                           Wz, Uz, bz.reshape(d),
                                           Wr, Ur, br.reshape(d),
                                           Wp, Up, bp.reshape(d),
                                           Sigma_y, Wy, by.reshape(yd))
        
        train_y_vec[i,j] = y


train_y = (np.sum(train_y_vec, axis = 0))/M


#Test predictions
#Mean weights
Wz,Uz,bz = update.extract_W_weights(EWz_bar, d, ud)
Wr,Ur,br = update.extract_W_weights(EWr_bar, d, ud)
Wp,Up,bp = update.extract_W_weights(EWp_bar, d, ud)
Wy,_,by = update.extract_W_weights(EWy_bar, d, 0)

h = Eh[-1,:,0]
u = y_last
test_y = np.zeros((T_new, yd))

for j in range(0, T_new):
    z, r, v, h, yt = gen.stoch_GRU_step_mean(h, u, Wz, Uz, bz.reshape(d),
                                        Wr, Ur, br.reshape(d),
                                        Wp, Up, bp.reshape(d),
                                        Wy, by.reshape(yd))
    test_y[j,:] = yt
    u = yt

ran = np.arange(stop, stop+T_new, dt)
plt.plot(ran, test_y, label = 'mean_test' )


    
test_y_vec = np.zeros((M,T_new))


for i in range(0,M):
    h = h_samples_vec[samples_len-M+i,-1,:]
    
    Wz_bar = Wz_bar_samples_vec[samples_len-M+i]
    Wz,Uz,bz = update.extract_W_weights(Wz_bar, d, ud)

    Wr_bar = Wr_bar_samples_vec[samples_len-M+i]
    Wr,Ur,br = update.extract_W_weights(Wr_bar, d, ud)

    Wp_bar = Wp_bar_samples_vec[samples_len-M+i]
    Wp,Up,bp = update.extract_W_weights(Wp_bar, d, ud)

    Wy_bar = Wy_bar_samples_vec[samples_len-M+i]
    Wy,_,by = update.extract_W_weights(Wy_bar, d, 0)
    
    u = y_last
    
    for j in range(0,T_new):
        
        z, r, v, h, y = gen.stoch_GRU_step(np.diag(1/inv_var), h, u,
                                           Wz, Uz, bz.reshape(d),
                                           Wr, Ur, br.reshape(d),
                                           Wp, Up, bp.reshape(d),
                                           Sigma_y, Wy, by.reshape(yd))
        
        test_y_vec[i,j] = y
        u = y


test_y = (np.sum(test_y_vec, axis = 0))/M




plt.plot(t[1:stop], train_y.reshape(T), label = 'sample_train')




plt.plot(ran, test_y, label = 'sample_test' )
plt.title('N={}, Train_Weights={}, d={}'.format(N,train_weights,d))
plt.legend()
plt.show()

