import numpy as np
from ssm import messages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys


def build_sys(T,d,info_args):
    J_ini, h_ini, log_z_ini, J_dyn_11, J_dyn_21, J_dyn_22, h_dyn_1, h_dyn_2, log_Z_dyn, J_obs, h_obs, log_Z_obs = info_args
    #Build precision
    prec = np.zeros((T*d,T*d))
    for t in range(0,T-1):
        prec[t*d:t*d+d, t*d:t*d+d] += J_obs[t]
        prec[t*d:t*d+d, t*d:t*d+d] += J_dyn_11[t]
        if t == 0:
            prec[t*d:t*d+d, t*d:t*d+d] += J_ini
        else:
            prec[t*d:t*d+d, t*d:t*d+d] += J_dyn_22[t-1]

        #cross term
        prec[t*d:t*d+d,t*d+d:t*d+2*d]  += J_dyn_21[t].T
        prec[t*d+d:t*d+2*d, t*d:t*d+d] += J_dyn_21[t]

    prec[(T-1)*d:(T-1)*d+d,(T-1)*d:(T-1)*d+d] += J_obs[T-1]+J_dyn_22[T-2] 

    #Build prec_muT
    val = np.zeros((T*d,1))
    val[:,0] += h_obs.ravel()
    val[:d,0] += h_ini.ravel()
    val[d:,0] += h_dyn_2.ravel()
    val[:(T-1)*d,0]+= h_dyn_1.ravel()

    covar = np.linalg.inv(prec)
    mu = covar @ val

    return mu, covar



T=3
d=2
N=1
U=1

args = messages.make_lds_parameters(T,d,N,U)

info_args = messages.convert_mean_to_info_args(*args)


M = 100000
samples1 = np.zeros((M,T*d))
for i in range(0,M):
    xs = messages.kalman_info_sample(*info_args)
    samples1[i] = xs.ravel()



df = pd.DataFrame(samples1)
col = ['mp']*M
df[6] = col

#sns.pairplot(df, diag_kind = 'kde')
#plt.show()



ll, smoothed_mus, smoothed_Sigmas, off = messages.kalman_info_smoother(*info_args)


mu, covar = build_sys(T,d,info_args)

samples2 = np.zeros((M,T*d))
for i in range(0,M):
    xs = np.random.multivariate_normal(mu[:,0], covar)
    samples2[i] = xs.ravel()



#data = {'Method':['mp', 'full'], 'values':[samples1, samples2]}    
#df = pd.DataFrame(data)
#print(df)
df2 = pd.DataFrame(samples2)
col = ['full']*M
df2[6] = col

df3 = df.append(df2)

sns.pairplot(df3, hue = 6,diag_kind = 'kde')
plt.show()


