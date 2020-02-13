import numpy as np
import var_updates as update
import build_model as build
import main_loop as loop
import generate_GRU as gen
import plot_dist as plot_dist
from scipy.special import expit
import  matplotlib.pyplot as plt
import os
import sys
import time

#seed = 4571#np.random.randint(0,10000)
seed = 4668
np.random.seed(seed)
print('random_seed:',seed)

'''
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
#u = data[:T].reshape(T,ud,1)
u = np.zeros((T,ud,1))
y = data[1:stop].reshape(T,yd,1)
y_last = y[-1,:,0]



T_full = len(data)-1
y_full = data[1:].reshape(T_full, ud, 1)

T_test = T_full-T
#u_test = data[stop-1:-1].reshape(T_test,ud,1)
y_test = data[stop:].reshape(T_test,yd,1)
#####################
'''



train_weights = True
#Loop parameters
N=1000
log_check = 1
N_burn = int(.9*N)
M = N-N_burn-1  #number of test samples should be less than N-N_burn
T_check = 3 


#Generate data 
T=10

d=10
ud = 1
yd = 1
alpha = 1.3
tau = .8

h0 = 0*np.ones(d)
u = np.zeros((T,ud,1))

var=.1
inv_var = np.ones(d)*1/var
var_y = .1
Sigma_y = var_y*np.identity(yd)
Sigma_y_inv = 1/var_y*np.identity(yd)



#Initialize weights
theta_var = (1/3)**2
Sigma_theta = np.ones((d,d+ud+1))*theta_var

theta_y_var = (1/3**2)
Sigma_y_theta = np.ones((yd,d+1))*theta_y_var


L=-.3
U= .4

#True Weights
WpT_bar,WpT,UpT,bpT,WpT_mu_prior  = update.init_weights(L,U, Sigma_theta,
                                                        d, ud)

WyT_bar,WyT,_,byT,WyT_mu_prior  = update.init_weights(L,U, Sigma_y_theta,
                                                      d, 0)

#Initialized Weights
L=-.9
U= .9

Wp_bar,Wp,Up,bp,Wp_mu_prior  = update.init_weights(L,U, Sigma_theta,
                                                        d, ud)
Wy_bar,Wy,_,by,Wy_mu_prior  = update.init_weights(L,U, Sigma_y_theta,
                                                      d, 0)






h = np.zeros((T+1,d,1))
y=  np.zeros((T,yd,1))
h[0,:,0] = h0
h_0 = h0
for j in range(0,T):
     _, ht, yt = gen.stoch_RNN_step(np.diag(1/inv_var),
                                          h_0, u[j,:,0],
                                          WpT, UpT, bpT.reshape(d),
                                          Sigma_y, WyT, byT.reshape(yd),
                                          alpha, tau)
     h[j+1,:,0] = ht
     h_0 = h[j+1,:,0]
     y[j,:,0] = yt
    


h_samples, v_samples,  Wp_bar_samples, Wy_bar_samples, h_samples_vec, Wp_bar_samples_vec, Wy_bar_samples_vec, h_plot_samples, log_joint_vec = loop.gibbs_loop(N, N_burn, T, d,
                                                 T_check, ud, yd, h0,
                                                 inv_var, Sigma_y_inv,
                                                 Sigma_theta, Sigma_y_theta,
                                                 Sigma_y,
                                                 Wp_mu_prior, Wy_mu_prior,
                                                 Wp, Up,
                                                 bp, Wy, by, train_weights,
                                                 u, y, h,log_check,                                                                      alpha, tau)


Eh = h_samples/(N-N_burn-1)
Ev = v_samples/(N-N_burn-1)
EWp_bar = Wp_bar_samples/(N-N_burn-1)
EWy_bar = Wy_bar_samples/(N-N_burn-1)

print('Eh')
print(Eh)
print('Ev')
print(Ev)
print('EWp_bar')
print(np.round(EWp_bar,4))
print('EWy_bar')
print(np.round(EWy_bar,4))

print('Wp_bar_True')
print(np.round(WpT_bar,4))
print('Wy_bar_True')
print(np.round(WyT_bar,4))



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
Wp,Up,bp = update.extract_W_weights(EWp_bar, d, ud)
Wy,_,by = update.extract_W_weights(EWy_bar, d, 0)


h = h0
train_y = np.zeros((T, yd))
train_y2 = np.zeros((T,yd))

for j in range(0, T):
    v, h, yt = gen.stoch_RNN_step_mean(h, u[j,:,0],
                                             Wp, Up, bp.reshape(d),
                                             Wy, by.reshape(yd), alpha, tau)
    train_y[j,:] = yt
    train_y2[j,:] = Wy @ Eh[j,:,0] + by.reshape(yd)


ran = np.arange(0,T)
plt.plot(ran,y.reshape(T), label = 'True')
plt.plot(ran, train_y.reshape(T), label = 'mean_train')


train_y_vec = np.zeros((M,T))
train_y_vec2 = np.zeros((M,T))


for i in range(0,M):
    h = h0
    h2 = h_samples_vec[-(i+1)]
    
    Wp_bar = Wp_bar_samples_vec[-(i+1)]
    Wp,Up,bp = update.extract_W_weights(Wp_bar, d, ud)

    Wy_bar = Wy_bar_samples_vec[-(i+1)]
    Wy,_,by = update.extract_W_weights(Wy_bar, d, 0)
    
    for j in range(0,T):
        v, h, y = gen.stoch_RNN_step(np.diag(1/inv_var), h, u[j,:,0],
                                           Wp, Up, bp.reshape(d),
                                           Sigma_y, Wy, by.reshape(yd),
                                           alpha, tau)
        
        
        train_y_vec[i,j] = y
        train_y_vec2[i,j] = Wy @ h2[j]+by.reshape(yd)


train_y = (np.sum(train_y_vec, axis = 0))/M
train_y2 = (np.sum(train_y_vec2,axis=0))/M

plt.plot(ran, train_y.reshape(T), label='sample_train')
plt.plot(ran, train_y2.reshape(T), label='Wyi@hi+byi_train')


'''
#Test predictions
#mean weights
Wp,Up,bp = update.extract_W_weights(EWp_bar, d, ud)
Wy,_,by = update.extract_W_weights(EWy_bar, d, 0)

h = Eh[-1,:,0]
u = np.zeros(yd)#y_last
test_y = np.zeros((T_new, yd))

for j in range(0, T_new):
    v, h, yt = gen.stoch_RNN_step_mean(h, u, 
                                        Wp, Up, bp.reshape(d),
                                             Wy, by.reshape(yd), alpha, tau)
    test_y[j,:] = yt
    u = np.zeros(yd)#yt

ran = np.arange(stop, stop+T_new, dt)
plt.plot(ran, test_y, label = 'mean_test' )

test_y_vec = np.zeros((M,T_new))


for i in range(0,M):
    h = h_samples_vec[-(i+1),-1,:]

    Wp_bar = Wp_bar_samples_vec[-(i+1)]
    Wp,Up,bp = update.extract_W_weights(Wp_bar, d, ud)

    Wy_bar = Wy_bar_samples_vec[-(i+1)]
    Wy,_,by = update.extract_W_weights(Wy_bar, d, 0)
    

    u = np.zeros(yd)#y_last
    
    for j in range(0,T_new):
        v, h, y = gen.stoch_RNN_step(np.diag(1/inv_var), h, u,
                                           Wp, Up, bp.reshape(d),
                                           Sigma_y, Wy, by.reshape(yd),
                                           alpha, tau)
        
        
        test_y_vec[i,j] = y
        u = np.zeros(yd)#y


test_y = (np.sum(test_y_vec, axis = 0))/M






plt.plot(ran, test_y, label='sample_test')
'''

plt.legend()
plt.title('N={}, M={}, Train_Weights={}, d={},alpha={},tau={},\n Var_h={}, Var_y={}'.format(N,M,train_weights,d,alpha, tau, var, var_y ))


plt.savefig(path+ '/d{}_N{}.png'.format( d, N))


plt.show()

