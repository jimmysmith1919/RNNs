import numpy as np
import var_updates as update
import build_model as build
import main_loop as loop
import generate_GRU as gen
import plot_dist as plot_dist
from scipy.special import expit
import scipy.stats as stats
import  matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import os
import time


#seed = np.random.randint(0,100000)
#print(seed)
#np.random.seed(seed)
#np.random.seed(24356)
#np.random.seed(67492)
#np.random.seed(88095)
np.random.seed(45771)

T= 5
d= 3
ud = 1
yd = 1
h0 = 0*np.ones(d)
#h0 = np.random.rand(d)
u = np.random.uniform(-1,1,size=(T,ud,1))
var=.1
inv_var = np.ones(d)*1/var


var_y = .1
Sigma_y = var_y*np.identity(yd)
Sigma_y_inv = 1/var_y*np.identity(yd)

alpha=1.3
tau=.8

#Initialize weights
theta_var = (1/3)**2
Sigma_theta = np.ones((d,d+ud+1))*theta_var

theta_y_var = (1/3**2)
Sigma_y_theta = np.ones((yd,d+1))*theta_y_var

L=-.9
U= .9


Wp_bar,Wp,Up,bp,Wp_mu_prior  = update.init_weights(L,U, Sigma_theta, d, ud)
Wy_bar,Wy,_,by,Wy_mu_prior  = update.init_weights(L,U, Sigma_y_theta, d, 0)


train_weights = True


### Get synthetic y's #################
M=100

y_vec = np.zeros((T,yd,1))

L=.3
U=.4


Wp_bar_true = Wp_mu_prior #np.random.uniform(L,U,size = (d,d+ud+1))
Wy_bar_true = Wy_mu_prior #np.random.uniform(L,U,size = (d,yd+1))



Wpy, Upy, bpy = update.extract_W_weights(Wp_bar_true, d, ud)
Wyy, _, byy = update.extract_W_weights(Wy_bar_true, d, 0)
#Wyy, _, byy = update.extract_W_weights(Wy_mu_prior, d, 0)


'''
Wzy, Uzy, bzy = update.extract_W_weights(Wz_mu_prior, d, ud)
Wry, Ury, bry = update.extract_W_weights(Wr_mu_prior, d, ud)
Wpy, Upy, bpy = update.extract_W_weights(Wp_mu_prior, d, ud)
Wyy, _, byy = update.extract_W_weights(Wy_mu_prior, d, 0)
'''


for i in range(0,M):
    h = h0
    for t in range(0,T):
        v, h, y = gen.stoch_RNN_step(np.diag(1/inv_var), h, u[t,:,0],
                                      Wpy, Upy, bpy.reshape(d),
                                      Sigma_y, Wyy, byy.reshape(yd),
                                      alpha, tau)
        y_vec[t,:,0] += y
    

y = y_vec/M

###############################################




#Initialize h
h = np.zeros((T+1,d,1))
h[0,:,0] = h0
for i in range(1,T+1):
    #TESTING
    h[i,:,0] = np.random.multivariate_normal(h[i-1,:,0], np.diag(1/inv_var) )
    #np.random.uniform(-.5,.5,3 
    


#Loop parameters
N = 1000000
M = 1000000
log_check = N
h_samples =0
v_samples =0
Wp_bar_samples = 0
Wy_bar_samples = 0

N_burn = int(.4*N)

T_check = 0
d_check = 1






h_samples, v_samples, Wp_bar_samples, Wy_bar_samples, h_samples_vec,  Wp_bar_samples_vec, Wy_bar_samples_vec, h_plot_samples, log_joint_vec = loop.gibbs_loop(N, N_burn, T, d,
                                                 T_check, ud, yd, h0,
                                                 inv_var, Sigma_y_inv,
                                                 Sigma_theta, Sigma_y_theta,
                                                 Sigma_y,
                                                 Wp_mu_prior, Wy_mu_prior, 
                                                 Wp, Up,
                                                 bp, Wy, by, train_weights,
                                                                                                                                                                                                                                                          u, y, h, log_check, alpha, tau)







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

print('Wp_bar_prior_mean')
print(np.round(Wp_mu_prior,4))
print('Wy_mu_prior_mean')
print(Wy_mu_prior)

print('Wp_bar_true')
print(Wp_bar_true)
print('Wy_bar_true')
print(Wy_bar_true)

v_vec = np.zeros((M,T,d,3))
h_vec = np.zeros((M,T,d))

Wp_bar_vec = np.zeros((M,d,d+ud+1))
Wy_bar_vec = np.zeros((M,yd,d+1))

for i in range(0,M):
    h = h0

    Wp_bar = np.random.normal(Wp_mu_prior, np.sqrt(Sigma_theta))
    Wp,Up,bp = update.extract_W_weights(Wp_bar, d, ud)

    Wy_bar = np.random.normal(Wy_mu_prior, np.sqrt(Sigma_y_theta))
    Wy,_,by = update.extract_W_weights(Wy_bar, d, 0)

    #for t in range(0,T_check):
    for t in range(0,T):
        v, h, y = gen.stoch_RNN_step(np.diag(1/inv_var), h, u[t,:,0],
                                           Wp, Up, bp.reshape(d),
                                           Sigma_y, Wy,
                                           by.reshape(yd), alpha, tau)

        h_vec[i,t,:] = h
        v_vec[i,t,:,:] = v
    Wp_bar_vec[i,:,:] = Wp_bar
    Wy_bar_vec[i,:,:] = Wy_bar
    

print('prior')
print('Ev')
print(np.sum(v_vec,axis=0)/M)
        
#check h
h_vec = h_vec.reshape(M,T*d)
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


'''
#check Wz
Wz_bar_vec = Wz_bar_vec.reshape(M,d*(d+ud+1))
Wz_bar_samples_vec = Wz_bar_samples_vec.reshape(N-N_burn-1,d*(d+ud+1))

df = pd.DataFrame(Wz_bar_vec)
col = ['prior']*M
df[d*(d+ud+1)]=col

df2 = pd.DataFrame(Wz_bar_samples_vec)
col = ['gibbs']*(N-N_burn-1)
df2[d*(d+ud+1)]=col

df3 = df.append(df2)

sns.pairplot(df3, hue=d*(d+ud+1), diag_kind = 'kde')
plt.savefig(path+'/Wz_N{}_M{}.png'.format(N,M))
plt.close()

#check Wr
Wr_bar_vec = Wr_bar_vec.reshape(M,d*(d+ud+1))
Wr_bar_samples_vec = Wr_bar_samples_vec.reshape(N-N_burn-1,d*(d+ud+1))

df = pd.DataFrame(Wr_bar_vec)
col = ['prior']*M
df[d*(d+ud+1)]=col

df2 = pd.DataFrame(Wr_bar_samples_vec)
col = ['gibbs']*(N-N_burn-1)
df2[d*(d+ud+1)]=col

df3 = df.append(df2)

sns.pairplot(df3, hue=d*(d+ud+1), diag_kind = 'kde')
plt.savefig(path+'/Wr_N{}_M{}.png'.format(N,M))
plt.close()

#check Wp
Wp_bar_vec = Wp_bar_vec.reshape(M,d*(d+ud+1))
Wp_bar_samples_vec = Wp_bar_samples_vec.reshape(N-N_burn-1,d*(d+ud+1))

df = pd.DataFrame(Wp_bar_vec)
col = ['prior']*M
df[d*(d+ud+1)]=col

df2 = pd.DataFrame(Wp_bar_samples_vec)
col = ['gibbs']*(N-N_burn-1)
df2[d*(d+ud+1)]=col

df3 = df.append(df2)

sns.pairplot(df3, hue=d*(d+ud+1), diag_kind = 'kde')
plt.savefig(path+'/Wp_N{}_M{}.png'.format(N,M))
plt.close()

#check Wy
Wy_bar_vec = Wy_bar_vec.reshape(M,yd*(d+1))
Wy_bar_samples_vec = Wy_bar_samples_vec.reshape(N-N_burn-1,yd*(d+1))

df = pd.DataFrame(Wy_bar_vec)
col = ['prior']*M
df[yd*(d+1)]=col

df2 = pd.DataFrame(Wy_bar_samples_vec)
col = ['gibbs']*(N-N_burn-1)
df2[yd*(d+1)]=col

df3 = df.append(df2)

sns.pairplot(df3, hue=yd*(d+1), diag_kind = 'kde')
plt.savefig(path+'/Wy_N{}_M{}.png'.format(N,M))
plt.close()
'''




'''
plt.hist(h_samples_vec[:,d_check].reshape(N-N_burn-1), bins=100, density=True)
plt.hist(h_vec[:,d_check], bins=100, histtype='step',color='r', density=True,
         label='prior')
plt.xlabel('h_t')
plt.ylabel('P(h_t|h_{t-1})')
plt.title('Gen: T={}, T_check={}, d_check={}, N={}, Var={}, var_y={}'.format(
    T,T_check,d_check,N,var, var_y))
plt.legend()
plt.show()
'''
sys.exit()



#print('y')
#print(y)

'''
Wy, _, by = update.extract_W_weights(EWy_bar, d, 0)
'''

#print('Wy_prior_mean')
#print(Wyy)

#print('by_prior_mean')
#print(byy)

'''
a = Wy @ Eh[1:] + by
print('y_approx_learned_weights')
print(a)
'''
'''
a = Wyy @ Eh[1:] + byy
print('y_approx_true_weights')
print(a)

y_vec = np.zeros((T,yd,1))

EWz, EUz, Ebz = update.extract_W_weights(EWz_bar, d, ud)
EWr, EUr, Ebr = update.extract_W_weights(EWr_bar, d, ud)
EWp, EUp, Ebp = update.extract_W_weights(EWp_bar, d, ud)


M= 10000
for i in range(0,M):
    h = h0
    for t in range(0,T):
        z, r, v, h, y = gen.stoch_GRU_step(np.diag(1/inv_var), h, u[t,:,0],
                                       EWz, EUz, Ebz.reshape(d),
                                       EWr, EUr, Ebr.reshape(d),
                                       EWp, EUp, Ebp.reshape(d),
                                           Sigma_y, Wyy, byy.reshape(yd))
        y_vec[t,:,0] += y
    

b = y_vec/M

print('y_approx_learn_weights')
print(b)
'''

      
'''
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
                                                   Sigma_y, Wy, by.reshape(yd))
    
    
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
print('Wz_bar_prior_mean')
print(np.round(Wz_mu_prior,4))
print('Wr_bar_prior_mean')
print(np.round(Wr_mu_prior,4))
print('Wp_bar_prior_mean')
print(np.round(Wp_mu_prior,4))
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

bin_vec = plot_dist.get_binary()
x = np.linspace(-2,2,1000)

marg_y = plot_dist.marginal_y(y, bin_vec, h0, var, Wzy, Uzy, bzy.reshape(d),
                             Wry, Ury, bry.reshape(d), Wpy, Upy,
                              bpy.reshape(d),
                              u, Wyy, byy.reshape(yd), var_y)

h_like,_,_,_,_ = plot_dist.get_likelihood(x,
                    bin_vec, h0, var,
                     Wzy, Uzy, bzy.reshape(d),
                     Wry, Ury, bry.reshape(d),
                     Wpy, Upy, bpy.reshape(d), u[T_check-1,:,0])


y_p = plot_dist.y_prior2(x, y, Wyy, byy, var_y)
yp = y_p[0,0,:]

post = (h_like*yp)/marg_y




plt.plot(x,post,'r',label = 'post')


M = 10000
post_vec = 0
c_vec = 0
for m in range(0,M):
    Wz_bar = np.random.normal(Wz_mu_prior, np.sqrt(Sigma_theta))
    Wzy,Uzy,bzy = update.extract_W_weights(Wz_bar, d, ud)
    
    Wr_bar = np.random.normal(Wr_mu_prior, np.sqrt(Sigma_theta))
    Wry,Ury,bry = update.extract_W_weights(Wr_bar, d, ud)
    
    Wp_bar = np.random.normal(Wp_mu_prior, np.sqrt(Sigma_theta))
    Wpy,Upy,bpy = update.extract_W_weights(Wp_bar, d, ud)

    Wy_bar = np.random.normal(Wy_mu_prior, np.sqrt(Sigma_y_theta))
    Wyy,_,byy = update.extract_W_weights(Wy_bar, d, 0)
    
    
    marg_y = plot_dist.marginal_y(y, bin_vec, h0, var, Wzy, Uzy,
                                  bzy.reshape(d),
                                  Wry, Ury, bry.reshape(d), Wpy, Upy,
                                  bpy.reshape(d),
                                  u, Wyy, byy.reshape(yd), var_y)

    h_like,_,_,_,_ = plot_dist.get_likelihood(x,
                                              bin_vec, h0, var,
                                              Wzy, Uzy, bzy.reshape(d),
                                              Wry, Ury, bry.reshape(d),
                                              Wpy, Upy, bpy.reshape(d),
                                              u[T_check-1,:,0])


    y_p = plot_dist.y_prior2(x, y, Wyy, byy, var_y)
    yp = y_p[0,0,:]

    
    post_vec += (h_like*yp)
    c_vec += marg_y


con = c_vec/M
post = post_vec/M
post1 = post/con
 
plt.plot(x,post1,'k',label = 'post1')

plt.hist(h_samples_vec[:,d_check].reshape(N-N_burn-1), bins=100, density=True)
plt.xlabel('h_t')
plt.ylabel('P(h_1|y_1)')
plt.title('Gen: T={}, T_check={}, d_check={}, N={}, Var={}, var_y={}'.format(
    T,T_check,d_check,N,var, var_y))
#plt.legend()
#plt.show()



'''
bin_vec = plot_dist.get_binary()
x = np.linspace(-2,2,1000)
'''



###
Wzy, Uzy, bzy = update.extract_W_weights(Wz_mu_prior, d, ud)
Wry, Ury, bry = update.extract_W_weights(Wr_mu_prior, d, ud)
Wpy, Upy, bpy = update.extract_W_weights(Wp_mu_prior, d, ud)

###

mix_pdf, mix_mean, pv_mean, pz, pr = plot_dist.get_likelihood(x,
                    bin_vec, h0, var,
                     Wzy, Uzy, bzy.reshape(d),
                     Wry, Ury, bry.reshape(d),
                     Wpy, Upy, bpy.reshape(d), u[T_check-1,:,0])

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


plt.plot(x, mix_pdf, 'g', label='prior')
plt.legend()
plt.show()

'''
plt.hist(h_samples_vec[:,0].reshape(N-N_burn-1), bins=100, density=True)
plt.xlabel('h_t')
plt.ylabel('P(h_t|h_{t-1})')
plt.title('Gibbs: T={}, T_check={}, N={}, Var={}'.format(
    T,T_check,N,var))
plt.legend()
plt.show()
'''

'''
M=10

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
                                           Wp, Up, bp.reshape(d),
                                           Sigma_y, Wy, by.reshape(yd))

    z_vec[i,:] = z
    r_vec[i,:] = r
    v_vec[i,:] = v
    h_vec[i,:] = h
'''


'''
EWz, EUz, Ebz = update.extract_W_weights(EWz_bar, d, ud)
EWr, EUr, Ebr = update.extract_W_weights(EWr_bar, d, ud)
EWp, EUp, Ebp = update.extract_W_weights(EWp_bar, d, ud)



for i in range(0,M):
    h = h0
    for t in range(0,T_check):
        z, r, v, h, y = gen.stoch_GRU_step(np.diag(1/inv_var), h, u[t,:,0],
                                           EWz, EUz, Ebz.reshape(d),
                                           EWr, EUr, Ebr.reshape(d),
                                           EWp, EUp, Ebp.reshape(d),
                                           Sigma_y, Wyy, byy.reshape(yd))

    z_vec[i,:] = z
    r_vec[i,:] = r
    v_vec[i,:] = v
    h_vec[i,:] = h
'''

'''
#plt.plot(x, mix_pdf, 'r', label='pdf')
plt.hist(h_samples_vec[:,d_check].reshape(N-N_burn-1), bins=100, density=True)
plt.hist(h_vec[:,d_check], bins=100, histtype='step',color='r', density=True, label='prior')
plt.xlabel('h_t')
plt.ylabel('P(h_t|h_{t-1})')
plt.title('Gen: T={}, T_check={}, d_check={}, N={}, Var={}, var_y={}'.format(
    T,T_check,d_check,N,var, var_y))
plt.legend()
plt.show()
''' 

'''
plt.plot(x, mix_pdf, 'r', label='pdf')
plt.hist(h_vec, bins=100, density=True)
plt.xlabel('h_t')
plt.ylabel('P(h_t|h_{t-1})')
plt.title('Samples from  Generative function')
plt.legend()
plt.show()
'''


