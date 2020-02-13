import numpy as np
import var_updates as update
import build_model as build
import v_froze_main_loop as loop
import generate_GRU as gen
import plot_dist as plot_dist
from scipy.special import expit
import  matplotlib.pyplot as plt
import os
import sys
import time
import itertools
import log_prob
from scipy.stats import norm


#seed = np.random.randint(0,10000)
#seed = 4667
#seed = 9882
#seed = 6467
#seed = 2903
#seed = 5106 
#seed = 2943
#seed = 7750
#seed = 2709
seed = 4050
np.random.seed(seed)
print('random_seed:',seed)



train_weights = True
#Loop parameters
N=10000
MM=1
log_check = 1
N_burn = int(.9*N)
M = N-N_burn-1  #number of test samples should be less than N-N_burn
T_check = 0 


#Generate data 
T=3
d=1
ud = 1
yd = 1
alpha = 2.3
tau_g = 1
tau = tau_g
h0 = 0*np.ones(d)
u = np.zeros((T,ud,1))

var= .3
inv_var = np.ones(d)*1/var
var_y = .1
Sigma_y = np.diag(np.random.uniform(size=yd))#var_y*np.identity(yd)
Sigma_y_inv = 1/var_y*np.identity(yd)



#Initialize weights
theta_var = .1
Sigma_theta = np.ones((d,d+ud+1))*theta_var

theta_y_var = .1
Sigma_y_theta = np.ones((yd,d+1))*theta_y_var


#L=-.9
#U= -.7

L=-.5
U=.5

#L = .3
#U = .9

#L = -.2
#U = .2

#True Weights
WpT_bar,WpT,UpT,bpT,WpT_mu_prior  = update.init_weights(L,U, Sigma_theta,
                                                        d, ud)



WyT_bar,WyT,_,byT,WyT_mu_prior  = update.init_weights(L,U, Sigma_y_theta,
                                                      d, 0)


##


theta_var = .1
Sigma_theta = np.ones((d,d+ud+1))*theta_var
##

#Initialized Weights
w_mu_prior = 0
Wp_mu_prior = w_mu_prior*np.ones((d,d+ud+1))
Wp_bar,Wp,Up,bp,_  = update.init_weights(L,U, Sigma_theta,
                                                        d, ud, Wp_mu_prior)

Wy_bar,Wy,_,by,Wy_mu_prior  = update.init_weights(L,U, Sigma_y_theta,
                                                      d, 0)


'''
Wp_bar = WpT_bar
Wp = WpT
Up = UpT
bp = bpT
Wp_mu_prior = WpT_mu_prior
'''

#Generate data
V = np.zeros((MM,T,d,1))
v = np.zeros((MM,T,d,3))
h = np.zeros((MM,T+1,d,1))
h[:,0,:,0] = h0
y = np.zeros((MM,T,yd,1))

ht = h[:,0]
for j in range(0,T):
     Vt,vt,ht,yt =  gen.stoch_RNN_step_vectorized(MM,d,1/inv_var,ht,
                                             u[j], WpT_bar,
                                             Sigma_y, WyT, byT,
                                             alpha, tau_g)
     V[:,j,:,:]= Vt
     v[:,j,:,:]  = vt
     h[:,j+1,:,:]= ht
     #y[:,j,:,:] = yt 

V_true = np.sum(V, axis=0)/MM
v_true = np.sum(v, axis=0)/MM
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
                                                                                                                                                                            u, y, h,log_check,                                                                      alpha, tau,v_true)


Eh = h_samples/(N-N_burn-1)
Ev = v_samples/(N-N_burn-1)
EWp_bar = Wp_bar_samples/(N-N_burn-1)
EWy_bar = Wy_bar_samples/(N-N_burn-1)

print('Eh')
print(Eh)
print('Ev')
print(Ev)
print('true_v')
print(v_true)
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

ran = np.arange(0,T+1)
#plt.plot(ran,h.reshape(T+1), label = 'True')

'''
#Training Predictions

#Mean weights
Wp,Up,bp = update.extract_W_weights(EWp_bar, d, ud)
Wy,_,by = update.extract_W_weights(EWy_bar, d, 0)


ht = h0
train_h = np.zeros((T, d))
train_h2 = np.zeros((T,d))

for j in range(0, T):
    v, ht, _ = gen.stoch_RNN_step_mean(ht, u[j,:,0],
                                             Wp, Up, bp.reshape(d),
                                             Wy, by.reshape(yd), alpha, tau)
    train_h[j,:] = ht
    #train_h2[j,:] = Wy @ Eh[j,:,0] + by.reshape(yd)



plt.plot(ran, train_h.reshape(T), label = 'mean_train')
'''



###

h_train = np.zeros((M,T+1,d,1))
h_train[:,0,:,0] = h0
h2_train = np.zeros((M,T+1,d,1))
h2_train[:,0,:,0] = h0
V2_train = np.zeros((M,T,d,1))


ht = h_train[:,0]



h_full = np.ones((M,T+1,d,1))*h




for j in range(0,T):
     newu = u[j]*np.ones((M,ud,1))
     ones = np.ones((M,1,1))
     
     #x = np.concatenate((ht, newu, ones), axis=1)
     x = np.concatenate((h_full[:,j], newu, ones), axis=1)
     fp = 2*(Wp_bar_samples_vec @ x)
     #fp = 2*(Wp_bar @ x)
     _,vt,ht,yt =  gen.stoch_RNN_step_vectorized(M,d,1/inv_var,ht,
                                             u[j], Wp_bar,
                                             Sigma_y, Wy, by,
                                             alpha, tau)
     
     
     h_train[:,j+1,:,:]= ht
     V = gen.big_V(M,v_samples_vec[:,j], alpha, tau, fp, d)     
     V = V.reshape(V.shape[0], V.shape[1],1)
     V2_train[:,j,:,:] = V

     h2_train[:,j+1,:,:] = 2*V-1
     
h_train = np.sum(h_train, axis=0)/M
h2_train = np.sum(h2_train, axis=0)/M
#V2_train = np.sum(V2_train, axis=0)/M



#######

def unnormalized_Wp_post(h,u, Wp, Up, bp, var, alpha, tau, T,d,
                         w_mu_prior,theta_var,v,K=3):
     
     fp = 2*(Wp*h[:-1,:,:]+Up*u + bp)
     scale=  np.sqrt(var)*np.ones(fp.shape)
     V1 = np.zeros(fp.shape)
     V2 = np.ones(fp.shape)
     V3 = 1/(2*alpha)*fp+1/2

          
     V = np.zeros(fp.shape)
     v1 = v[:,:,0].reshape(T,d,1)*np.ones(fp.shape)
     V += v1*V1
     v2 = v[:,:,1].reshape(T,d,1)*np.ones(fp.shape)
     V += v2*V2
     v3 = v[:,:,2].reshape(T,d,1)*np.ones(fp.shape)
     V += v3*V3
     

     
     mu = 2*V-1
     
     h_input = h[1:]*np.ones(fp.shape)
     h_pdf = norm.pdf(h_input, mu, scale)
     
     zeta1 = (-fp-alpha)/tau
     zeta2 = (fp-alpha)/tau
     
     pv1 = expit(zeta1)
     pv2 = np.exp(np.log(expit(zeta2))+np.log(expit(-zeta1)))
     pv3 = 1-(pv1+pv2)
     
     pv = v1*pv1 + v2*pv2 + v3*pv3

     hv = h_pdf*pv 
     
     log_unnorm = np.sum(np.log(hv), axis=0)

     scale = np.sqrt(theta_var)*np.ones(fp.shape[1:])
     log_w = norm.logpdf(Wp, w_mu_prior, scale)
     #log_U = norm.logpdf(Up[0], w_mu_prior, np.sqrt(theta_var))
     log_b = norm.logpdf(bp, w_mu_prior, scale)
     
     #return  np.exp(log_unnorm+log_w+log_U+log_b)
     return  np.exp(log_unnorm+log_w+log_b)



def transform(L, H, size, x, y):
    scalex = (x-L)/(H-L)
    newx = scalex*size

    scaley = (H-y)/(H-L)
    newy = scaley*size
    return newx, newy

     
print(seed)     
low = -1.5
high = 1.5
size=50
w=b=np.linspace(low,high,size)
W,B = np.meshgrid(w,b)     
B = np.flip(B, axis=0)  

unnorm = 0
UM = 100
for i in range(UM):
     Up = norm.rvs(w_mu_prior, np.sqrt(theta_var)).reshape(Up.shape)
     unnorm += unnormalized_Wp_post(h,u,W,Up,B,var,alpha,tau,T,d, w_mu_prior, theta_var, v_true)
     print(i)

unnorm = unnorm/UM


plt.imshow(unnorm, cmap='jet', interpolation='nearest')



for i in range(0,M):
     new_w, new_b = transform(low, high,size, Wp_bar_samples_vec[i,0,0],
                              Wp_bar_samples_vec[i,0,2])  
     plt.plot(new_w, new_b, '.', alpha=.7)

plt.contour(unnorm,zorder=M+10)

new_w, new_b = transform(low,high, size, WpT_bar[0,0],WpT_bar[0,2])
plt.plot(new_w, new_b, 'go', label='obs_sample')
new_w, new_b = transform(low,high, size, WpT_mu_prior[0,0],WpT_mu_prior[0,2])
plt.plot(new_w, new_b, 'gx', label='true_mean')
new_w, new_b = transform(low,high, size, 0,0)
plt.plot(new_w, new_b, 'rx', label='0,0')


plt.legend()
plt.xlabel('Wp')
plt.ylabel('bp')
plt.title('True unnormalized posterior vs gibbs samples')
plt.savefig(path+ '/d{}_N{}_W.png'.format( d, N))
#plt.show()
plt.close()








sys.exit()

plt.plot(ran, h_train.reshape(T+1), label='sample_train')
plt.plot(ran, h2_train.reshape(T+1), label='2*Vi(Wp_i)-1')
#plt.plot(ran[1:], V2_train.reshape(T), label='Vi')


####
'''
train_h_vec = np.zeros((M,T))
train_h_vec2 = np.zeros((M,T))


for i in range(0,M):
    ht = h0

    v2 = v_samples_vec[-(i+1)]
    
    Wp_bar = Wp_bar_samples_vec[-(i+1)]
    Wp,Up,bp = update.extract_W_weights(Wp_bar, d, ud)

    Wy_bar = Wy_bar_samples_vec[-(i+1)]
    Wy,_,by = update.extract_W_weights(Wy_bar, d, 0)
    
    for j in range(0,T):
         fp = 2*(Wp @ ht + Up @ u[j,:,0] + bp.reshape(d))
         v, ht, _ = gen.stoch_RNN_step(np.diag(1/inv_var), ht, u[j,:,0],
                                           Wp, Up, bp.reshape(d),
                                           Sigma_y, Wy, by.reshape(yd),
                                           alpha, tau)
        
        
         train_h_vec[i,j] = ht
         V = gen.big_V(1, v2[j].reshape(1,d,3), alpha, tau, fp, d)
         train_h_vec2[i,j] = 2*V-1
        


train_h = (np.sum(train_h_vec, axis = 0))/M
train_h2 = (np.sum(train_h_vec2,axis=0))/M

#plt.plot(ran, train_h.reshape(T), label='sample_train')
plt.plot(ran, train_h2.reshape(T), label='2Vi(Wp_i)-1_2')
'''

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




