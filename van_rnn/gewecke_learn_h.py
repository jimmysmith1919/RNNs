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
import pandas as pd
import seaborn as sns
import scipy.stats as stats

#seed = 4571#np.random.randint(0,10000)
seed = 4667
#seed = 4668
np.random.seed(seed)
print('random_seed:',seed)



train_weights = False
#Loop parameters
N=100000
MM=100
log_check = 1
N_burn = int(.5*N)
M = 100000#N-N_burn-1  #number of test samples should be less than N-N_burn
T_check = 3 


#Generate data 
T=5
d=1
ud = 1
yd = 1
alpha = 1.3
tau = .8

h0 = 0*np.ones(d)
u = np.zeros((T,ud,1))

var=.1
inv_var = np.random.uniform(size=d)#np.ones(d)*1/var
var_y = .1
Sigma_y = np.diag(np.random.uniform(size=yd))#var_y*np.identity(yd)
Sigma_y_inv = 1/var_y*np.identity(yd)



#Initialize weights
theta_var = .01#(1/3)**2
Sigma_theta = np.ones((d,d+ud+1))*theta_var

theta_y_var = .01#(1/3**2)
Sigma_y_theta = np.ones((yd,d+1))*theta_y_var


L=-.9
U= -.4

#L = .3
#U = .9

#L = -.2
#U = .2

#True Weights
WpT_bar,WpT,UpT,bpT,WpT_mu_prior  = update.init_weights(L,U, Sigma_theta,
                                                        d, ud)

WpT_bar = WpT_mu_prior
WpT, UpT, bpT = update.extract_W_weights(WpT_bar, d, ud)

WyT_bar,WyT,_,byT,WyT_mu_prior  = update.init_weights(L,U, Sigma_y_theta,
                                                      d, 0)
WyT_bar = WyT_mu_prior
WyT, _, byT = update.extract_W_weights(WyT_bar, d, 0)



#Initialized Weights
L=-.9
U= .9

#Wp_bar,Wp,Up,bp,Wp_mu_prior  = update.init_weights(L,U, Sigma_theta,
#                                                        d, ud)
Wy_bar,Wy,_,by,Wy_mu_prior  = update.init_weights(L,U, Sigma_y_theta,
                                                      d, 0)
Wp_bar = WpT_bar
Wp = WpT
Up = UpT
bp = bpT
Wp_mu_prior = WpT_mu_prior


#Generate data
#v = np.zeros((MM,T,d,3))
h = np.zeros((MM,T+1,d,1))
h[:,0,:,0] = h0
y = np.zeros((MM,T,yd,1))

ht = h[:,0]
for j in range(0,T):
     vt,ht,yt =  gen.stoch_RNN_step_vectorized(MM,d,1/inv_var,ht,
                                             u[j], WpT_bar,
                                             Sigma_y, WyT, byT,
                                             alpha, tau)
     #v[:,j,:,:]  = vt
     h[:,j+1,:,:]= ht
     #y[:,j,:,:] = yt 

#v = np.sum(v, axis=0)/MM
h = np.sum(h, axis=0)/MM
y = np.sum(y, axis =0)/MM




####For h observed
Wy_bar = np.zeros(Wy_bar.shape)
Wy_mu_prior = np.zeros(Wy_mu_prior.shape)
Wy = np.zeros(WyT.shape)
by = np.zeros(byT.shape)
####

h_samples, v_samples,  Wp_bar_samples, Wy_bar_samples, h_samples_vec,v_samples_vec, Wp_bar_samples_vec, Wy_bar_samples_vec, h_plot_samples, log_joint_vec = loop.gibbs_loop(N, N_burn, T, d,
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

h_train = np.zeros((M,T+1,d,1))
h_train[:,0,:,0] = h0
ht = h_train[:,0]

v_train = np.zeros((M,T,d,3))


for j in range(0,T):
     vt,ht,yt =  gen.stoch_RNN_step_vectorized(M,d,1/inv_var,ht,
                                             u[j], Wp_bar,
                                             Sigma_y, Wy, by,
                                             alpha, tau)
     
     
     h_train[:,j+1,:,:]= ht
     v_train[:,j,:,:]  = vt
     

Ev = np.sum(v_train, axis=0)/M


print('prior')
print('Ev')
print(Ev)

#check h
print(h_train.shape)
print(h_samples_vec.shape)

h_vec = h_train[:,1:].reshape(M,T*d)
h_samples_vec = h_samples_vec.reshape(N-N_burn-1,T*d)

df = pd.DataFrame(h_vec)
col = ['prior']*M
df[T*d]=col

df2 = pd.DataFrame(h_samples_vec)
col = ['gibbs']*(N-N_burn-1)
df2[T*d]=col

df3 = df.append(df2)

timestamp = time.time()
path = 'images/{}'.format(timestamp)
os.mkdir(path)


for i in range(0,T*d):
    print(stats.ks_2samp(h_samples_vec[:,i], h_vec[:,i]))


sns.pairplot(df3, hue=T*d, diag_kind = 'kde')
plt.savefig(path+'/h_N{}_M{}.png'.format(N,M))
plt.close()




