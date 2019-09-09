import numpy as np
from scipy.special import expit
import sys
import time
import matplotlib.pyplot as plt
from PG_int import qdf_log_pdf, entropy_q, qdf_log_pdf_vec, entropy_q_vec
from scipy import integrate
import os


def generate(T,d, yd, u, c0, h0,
             Wy, Wi, Wf, Wp, Wo, 
             Uy, Ui, Uf, Up, Uo, 
             by, bi, bf, bp, bo):
    cg = np.zeros((T,d))
    hg = np.zeros((T,d))
    v = np.zeros((T,d))
    zi = np.zeros((T,d))
    zf = np.zeros((T,d))
    zp = np.zeros((T,d))
    zo = np.zeros((T,d))
    yg = np.zeros((T,yd))

    

    zi[0,:] = expit(Wi @ h0 + Ui @ u[0,:] + bi) 
    zf[0,:] = expit(Wf @ h0 + Uf @ u[0,:] + bf) 
    zp[0,:] = expit(2*(Wp @ h0 + Up @ u[0,:] + bp))
    zo[0,:] = expit(Wo @ h0 + Uo @ u[0,:] + bo) 

    

    cg[0,:] = zf[0,:]*c0+zi[0,:]*(2*zp[0,:]-1) 
    print('cg')
    print(cg)
    
    v[0,:] =  expit(2*cg[0,:])  
    hg[0,:] = zo[0,:]*(2*v[0,:]-1)  
    yg[0,:] = Wy @ hg[0,:] + Uy @ u[0,:] + by
    
    
    for t in range(1,T):
        zi[t,:] = expit(Wi @ hg[t-1,:]+Ui @ u[t,:]+bi)  
        zf[t,:] = expit(Wf @ hg[t-1,:]+Uf @ u[t,:]+bf)  
        zp[t,:] = expit(2*(Wp @ hg[t-1,:]+Up @ u[t,:]+bp)) 
        zo[t,:] = expit(Wo @ hg[t-1,:]+Uo @ u[t,:]+bo) 
        cg[t,:] = zf[t,:]*cg[t-1,:]+zi[t,:]*(2*zp[t,:]-1) 

        v[t,:]  = expit(2*cg[t,:]) 
        hg[t,:] = zo[t,:]*(2*v[t,:]-1) 
        yg[t,:] = Wy @ hg[t,:] + Uy @ u[t,:] + by 
        
    return yg, cg, hg, v, zi, zf, zp, zo


def LSTM(c0, h0, Wi, Wf, Wp, Wo, bi, bf, bp, bo, Wy, by):
    i = expit(Wi @ h0 + bi)
    f = expit(Wf @ h0 + bf)
    p = np.tanh(Wp @ h0 + bp)
    o = expit(Wo @ h0 + bo)

    c = f*c0 + i*p
    h = o*np.tanh(c)
    y = Wy @ h + by
    return y,c,h,i,f,p,o




#Sine Wave#
end = 200
dt = 1
T_new = int(.2*end/dt)+200 #Number of new steps to predict

t = np.arange(0, end, dt)
data = np.sin((.06+.006)*t)


ud = 1 #u dimension
yd = 1 #y dimension

#T=len(data)-1
T_full=len(data)-1


#u = data[:-1].reshape(T,ud,1)
#y = data[1:].reshape(T,yd,1)
u_full = data[:-1].reshape(T_full,ud,1)

y_full = data[1:].reshape(T_full,ud,1)

stop = int(.8*end/dt)#int(3*len(u_full)/4)+1

T = stop-1
u = data[:stop-1].reshape(T,ud,1)
y = data[1:stop].reshape(T,yd,1)

T_test = T_full-T
u_test = data[stop-1:-1].reshape(T_test,ud,1)
y_test = data[stop:].reshape(T_test,yd,1)


#######################################################################


#Hyperparameters ##################################################3
d =3     #dimension of h and c
var_c = .01
var_h = .01
N=1

wfile = 'weights/'+ 'LSTM_d3_eps10_lr0.001_end200_1567747888.587923.npy'

#Initialize c 
inv_covar_c = 1/var_c*np.ones((d,1))
Sig_c = 1/inv_covar_c
mu_c0 = 0#np.random.uniform(-1,1) 
print('c0:',mu_c0)
c_0 = mu_c0*np.ones((d,1))

#Initialize h
inv_covar_h = 1/var_h*np.ones((d,1))
Sig_h = 1/inv_covar_h
mu_h0 = 0#np.random.uniform(-1,1) 
print('h0:',mu_h0)
h_0 = mu_h0*np.ones((d,1))


#Initialize Parameters
var_y = np.random.uniform(.001,.5)
Sigma_y = var_y*np.identity(yd)
inv_covar_y = 1/var_y*np.identity(yd)



weights = np.load(wfile)

Ui = np.zeros((d,1))#weights[0][0,:d].reshape(d,ud)   
Uf = np.zeros((d,1))#weights[0][0,d:2*d].reshape(d,ud)       
Up = np.zeros((d,1))#weights[0][0,2*d:3*d].reshape(d,ud) 
Uo = np.zeros((d,1))#weights[0][0,3*d:4*d].reshape(d,ud) 

#Wi = weights[1][:,:d] 
#Wf = weights[1][:,d:2*d]                                                       
#Wp = weights[1][:,2*d:3*d]
#Wo = weights[1][:,3*d:4*d]                                                   
                                                                               
#bi = weights[2][:d].reshape(d,1)
#bf = weights[2][d:2*d].reshape(d,1)
#bp = weights[2][2*d:3*d].reshape(d,1) 
#bo = weights[2][3*d:4*d].reshape(d,1) 

#Wy = weights[3].reshape(yd,d)                                                 
#by = weights[4].reshape(yd,1) 
Uy = np.zeros((yd,ud))


Wi = np.random.uniform(-1,1, size=(d,d))
Wf = np.random.uniform(-1,1, size=(d,d))
Wp = np.random.uniform(-1,1, size=(d,d))
Wo = np.random.uniform(-1,1, size=(d,d))
Wy = np.random.uniform(-1,1, size=(1,d))


bi = np.random.uniform(-1,1, size=(d,1))
bf = np.random.uniform(-1,1, size=(d,1))
bp = np.random.uniform(-1,1, size=(d,1))
bo = np.random.uniform(-1,1, size=(d,1))
by = np.random.uniform(-1,1, size=(1,1))

##Need to change code!
#Uy = np.random.uniform(-1,1, size=(yd,ud))










        
y_tr_gen= np.random.uniform(-1,1, size=(1,1))
cz = np.random.uniform(-1,1, size=(d,1))
hz = np.random.uniform(-1,1, size=(d,1))

'''
f = expit(Wf @ h_0 + (Uf*.06595209).reshape(d) + bf.reshape(d))
i = expit(Wi @ h_0 + (Ui*.06595209).reshape(d) + bi.reshape(d))
p = np.tanh(Wp @ h_0 + (Up*.06595209).reshape(d) + bp.reshape(d))
o = expit(Wo @ h_0 + (Uo*.06595209).reshape(d) + bo.reshape(d))

c = f*c_0+i*p
print('c')
print(c)
h = o*np.tanh(c)
print('h')
print(h)
'''

y_tr_gen, c_0, h_0, _,_,_,_,_ = generate(1,d, yd, y_tr_gen, 
                                         cz.reshape(d), 
                                         hz.reshape(d), 
                                         Wy, Wi, Wf, Wp, Wo, 
                                         Uy, Ui, Uf, Up, Uo, 
                                         by.reshape(yd), bi.reshape(d), 
                                         bf.reshape(d), bp.reshape(d), 
                                         bo.reshape(d))
print('y1')
print(y_tr_gen)

y,c,h,i,f,p,o = LSTM(cz.reshape(d), hz.reshape(d), Wi, Wf, Wp, Wo, 
                     bi.reshape(d), bf.reshape(d), bp.reshape(d), 
                     bo.reshape(d), Wy, by.reshape(1))

print('y')
print(y)



