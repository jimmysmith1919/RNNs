import numpy as np
import var_updates as update
import build_model as build
import main_loop_debug as loop
import generate_GRU as gen
import plot_dist as plot_dist
from scipy.special import expit
import  matplotlib.pyplot as plt
import os
import sys
import time

seed = 4571#np.random.randint(0,10000)
np.random.seed(seed)
print('random_seed:',seed)


## DATA  #######
#Sine Wave#
end = 200
dt = 1
T_new = 300


t = np.arange(0, end, dt)
#data = np.sin((.05)*t)
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
var=.1
inv_var = np.ones(d)*1/var
var_y = .001
Sigma_y = var_y*np.identity(yd)
Sigma_y_inv = 1/var_y*np.identity(yd)

#Initialize weights
theta_var = (1/3)**2
Sigma_theta = np.ones((d,d+ud+1))*theta_var

theta_y_var = (1/3**2)
Sigma_y_theta = np.ones((yd,d+1))*theta_y_var


L=-.9
U= .9

Wz_bar,Wz,Uz,bz,Wz_mu_prior = update.init_weights(L,U, Sigma_theta, d, ud)
Wr_bar,Wr,Ur,br,Wr_mu_prior  = update.init_weights(L,U, Sigma_theta, d, ud)
Wp_bar,Wp,Up,bp,Wp_mu_prior  = update.init_weights(L,U, Sigma_theta, d, ud)

Wy_bar,Wy,_,by,Wy_mu_prior  = update.init_weights(L,U, Sigma_y_theta, d, 0)


train_weights = True
#Loop parameters
N=1000000
log_check = N
N_burn = int(.999*N)
M = N-N_burn-1  #number of test samples should be less than N-N_burn
T_check = 100 
d_check = 5

##Load Trained Tensorflow Weights
#####
'''
wfile = 'weights/'+'GRU_d10_eps150_lr0.0001_end200_1577986258.269421.npy'
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
'''
#####




#Initialize h
h = np.zeros((T+1,d,1))
h[0,:,0] = h0
h_0 = h0
r = np.zeros((T,d,1))
z = np.zeros((T,d,1))
for j in range(0,T):
    zt, rt, _, ht, _ = gen.stoch_GRU_step(np.diag(1/inv_var),
                                          h_0, u[j,:,0],
                                          Wz, Uz, bz.reshape(d),
                                          Wr, Ur, br.reshape(d),
                                          Wp, Up, bp.reshape(d),
                                          Sigma_y, Wy, by.reshape(yd))
    r[j,:,0] = rt
    z[j,:,0] = zt
    h[j+1,:,0] = ht
    h_0 = h[j+1,:,0]
    
rh = np.zeros((T+1,d,1))




h_samples, z_samples, r_samples, v_samples, Wz_bar_samples,Wr_bar_samples, Wp_bar_samples, Wy_bar_samples, h_samples_vec, Wz_bar_samples_vec, Wr_bar_samples_vec, Wp_bar_samples_vec, Wy_bar_samples_vec, h_plot_samples, log_joint_vec,  z_samples_vec, r_samples_vec, v_samples_vec, h_samples_vec2 = loop.gibbs_loop(N, N_burn, T, d,
                                                 T_check, ud, yd, h0,
                                                 inv_var, Sigma_y_inv,
                                                 Sigma_theta, Sigma_y_theta,
                                                 Sigma_y,
                                                 Wz_mu_prior, Wr_mu_prior,
                                                 Wp_mu_prior, Wy_mu_prior, Wz,
                                                 Uz, bz, Wr, Ur, br, Wp, Up,
                                                 bp, Wy, by, train_weights,
                                                u, y, h, r, rh, z, log_check)

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
print(Ez[:100])
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

timestamp = time.time()
path = 'images/{}'.format(timestamp)
os.mkdir(path)

###
#plot evolution of parameters over time
plt.plot(h_plot_samples)
plt.savefig(path+'/samples')
plt.title('N={}, M={}, Train_Weights={}, d={}, Var_h={}, Var_y={}'.format(N,M,train_weights,d,var, var_y ))
plt.close()
for j in range(0,d):
    plt.plot(h_plot_samples[:,j])
    plt.savefig(path+'/samples_{}'.format(j))
    plt.close()

plt.plot(log_joint_vec)
plt.savefig(path+'/log_joint')
plt.close()
###


#Training Predictions

#Mean weights
Wz,Uz,bz = update.extract_W_weights(EWz_bar, d, ud)
Wr,Ur,br = update.extract_W_weights(EWr_bar, d, ud)
Wp,Up,bp = update.extract_W_weights(EWp_bar, d, ud)
Wy,_,by = update.extract_W_weights(EWy_bar, d, 0)


h = h0
train_y = np.zeros((T, yd))
train_y2 = np.zeros((T,yd))

for j in range(0, T):
    z, r, v, h, yt = gen.stoch_GRU_step_mean(h, u[j,:,0],
                                             Wz, Uz, bz.reshape(d),
                                             Wr, Ur, br.reshape(d),
                                             Wp, Up, bp.reshape(d),
                                             Wy, by.reshape(yd))
    train_y[j,:] = yt
    train_y2[j,:] = Wy @ Eh[j,:,0] + by.reshape(yd)



plt.plot(t[1:],y_full.reshape(T_full), label = 'True')
plt.plot(t[1:stop], train_y.reshape(T), label = 'mean_train')
plt.plot(t[1:stop], train_y2.reshape(T), label = 'EWy@Eh+Eby_train')

#samples_len = len(h_samples_vec[:,0,0])

train_y_vec = np.zeros((M,T))
train_y_vec2 = np.zeros((M,T))

print(Wz_bar_samples_vec.shape)


for i in range(0,M):
    h = h0
    h2 = h_samples_vec[-(i+1)]
    
    Wz_bar = Wz_bar_samples_vec[-(i+1)]
    Wz,Uz,bz = update.extract_W_weights(Wz_bar, d, ud)

    Wr_bar = Wr_bar_samples_vec[-(i+1)]
    Wr,Ur,br = update.extract_W_weights(Wr_bar, d, ud)

    Wp_bar = Wp_bar_samples_vec[-(i+1)]
    Wp,Up,bp = update.extract_W_weights(Wp_bar, d, ud)

    Wy_bar = Wy_bar_samples_vec[-(i+1)]
    Wy,_,by = update.extract_W_weights(Wy_bar, d, 0)
    
    for j in range(0,T):
        z, r, v, h, y = gen.stoch_GRU_step(np.diag(1/inv_var), h, u[j,:,0],
                                           Wz, Uz, bz.reshape(d),
                                           Wr, Ur, br.reshape(d),
                                           Wp, Up, bp.reshape(d),
                                           Sigma_y, Wy, by.reshape(yd))
        
        
        train_y_vec[i,j] = y
        train_y_vec2[i,j] = Wy @ h2[j]+by.reshape(yd)


train_y = (np.sum(train_y_vec, axis = 0))/M
train_y2 = (np.sum(train_y_vec2,axis=0))/M

plt.plot(t[1:stop], train_y.reshape(T), label='sample_train')
plt.plot(t[1:stop], train_y2.reshape(T), label='Wyi@hi+byi_train')

plt.legend()
plt.title('N={}, M={}, Train_Weights={}, d={}, Var_h={}, Var_y={}'.format(N,M,train_weights,d,var, var_y ))


plt.savefig(path+ '/d{}_N{}.png'.format( d, N))

plt.close()

'''
#Test predictions
#mean weights
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
    h = h_samples_vec[-(i+1),-1,:]
    
    Wz_bar = Wz_bar_samples_vec[-(i+1)]
    Wz,Uz,bz = update.extract_W_weights(Wz_bar, d, ud)

    Wr_bar = Wr_bar_samples_vec[-(i+1)]
    Wr,Ur,br = update.extract_W_weights(Wr_bar, d, ud)

    Wp_bar = Wp_bar_samples_vec[-(i+1)]
    Wp,Up,bp = update.extract_W_weights(Wp_bar, d, ud)

    Wy_bar = Wy_bar_samples_vec[-(i+1)]
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




plt.plot(ran, test_y, label='sample_test')

plt.legend()
plt.title('N={}, M={}, Train_Weights={}, d={}, Var_h={}, Var_y={}'.format(N,M,train_weights,d,var, var_y ))


plt.savefig(path+ '/d{}_N{}.png'.format( d, N))

plt.close()
'''


h_calc = (1-z_samples_vec)*h_samples_vec2[:,:-1]+z_samples_vec*(2*v_samples_vec-1)
print('calc')
print(h_calc[-20:,-5:,0])
print('z')
print(z_samples_vec[-20:,-5:,0])
print('hmin')
print(h_samples_vec2[-20:,-5:,0])
print('v')
print(v_samples_vec[-20:,-5:,0])

h_calc = np.sum(h_calc, axis=0)/M


h_calc_mean = (1-Ez)*Eh[:-1]+Ez*(2*Ev-1)



f_vec = np.zeros((M,T,d))
fr_vec = np.zeros((M,T,d))
fp_vec = np.zeros((M,T,d))
for j in range(0,M):
    x = np.concatenate((h_samples_vec2[j,:-1,:], u.reshape(T,ud), np.ones((T,1))), axis=1)
    xr = np.concatenate((r_samples_vec[j]*h_samples_vec2[j,:-1,:], u.reshape(T,ud), np.ones((T,1))), axis=1)

    f = Wz_bar_samples_vec[j] @ x.reshape(T,d+ud+1,1)
    fr = Wr_bar_samples_vec[j] @ x.reshape(T,d+ud+1,1)
    fp = Wp_bar_samples_vec[j] @ xr.reshape(T,d+ud+1,1)
    f_vec[j] = f[:,:,0]
    fr_vec[j] = fr[:,:,0]
    fp_vec[j] = fp[:,:,0]

sig_f_vec= expit(f_vec)
sig_fr_vec= expit(fr_vec)
sig_fp_vec= expit(2*fp_vec)

Ez2 = np.sum(sig_f_vec, axis=0)/M
Er2 = np.sum(sig_fr_vec, axis=0)/M
Ev2 = np.sum(sig_fp_vec, axis=0)/M




#plt.plot(t[1:],y_full.reshape(T_full), label = 'True')

for j in range(0,d):
    plt.plot(t[1:stop], Eh[1:,j], label='Eh')
    plt.plot(t[1:stop], h_calc[:,j], label='(1-z)h_mint+z(2v-1)', )
    #plt.plot(t[1:stop], h_calc_mean[:,j,0], label = '(1-ez)Eh_mint+Ez(2Ev-1)')
    plt.legend()
    plt.title('N={}, M={}, Train_Weights={}, d={}, Var_h={}, Var_y={}'.format(N,M,train_weights,d,var, var_y ))
    plt.savefig(path+ '/hd{}.png'.format( j))
    plt.close()

for j in range(0,d):
    plt.plot(t[1:stop], Ez[:,j,0], label = 'Ez')                           
    plt.plot(t[1:stop], Ez2[:,j], label = 'Ez_prior')                         
    plt.legend()
    plt.title('N={}, M={}, Train_Weights={}, d={}, Var_h={}, Var_y={}'.format(N,M,train_weights,d,var, var_y ))
    plt.savefig(path+ '/zd{}.png'.format( j))
    plt.close()



for j in range(0,d):
    plt.plot(t[1:stop], Er[:,j,0], label = 'Er')
    plt.plot(t[1:stop], Er2[:,j], label = 'Er_prior')
    plt.legend()
    plt.title('N={}, M={}, Train_Weights={}, d={}, Var_h={}, Var_y={}'.format(N,M,train_weights,d,var, var_y ))
    plt.savefig(path+ '/rd{}.png'.format( j))
    plt.close()



for j in range(0,d):
    plt.plot(t[1:stop], Ev[:,j,0], label = 'Ev')                             
    plt.plot(t[1:stop], Ev2[:,j], label = 'Ev_prior')
    plt.legend()
    plt.title('N={}, M={}, Train_Weights={}, d={}, Var_h={}, Var_y={}'.format(N,M,train_weights,d,var, var_y ))
    plt.savefig(path+ '/vd{}.png'.format( j))
    plt.close()

    

