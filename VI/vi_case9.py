import numpy as np
from scipy.special import expit
import sys
import time
import matplotlib.pyplot as plt
from PG_int import qdf_log_pdf, entropy_q, qdf_log_pdf_vec, entropy_q_vec
from scipy import integrate
import os


def generate(T,d, yd, u, c0, Sigma_c, h0, Sigma_h, Sigma_y,
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
d =10     #dimension of h and c
var_c = .01
var_h = .01
N=1

wfile = 'weights/'+ 'LSTM_d10_eps10_lr0.001_end200_1567782160.6545112.npy'

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

Ui = weights[0][0,:d].reshape(d,ud)   
Uf = weights[0][0,d:2*d].reshape(d,ud)       
Up = weights[0][0,2*d:3*d].reshape(d,ud) 
Uo = weights[0][0,3*d:4*d].reshape(d,ud) 

Wi = weights[1][:,:d] 
Wf = weights[1][:,d:2*d]                                                       
Wp = weights[1][:,2*d:3*d]
Wo = weights[1][:,3*d:4*d]                                                   
                                                                               
bi = weights[2][:d].reshape(d,1)
bf = weights[2][d:2*d].reshape(d,1)
bp = weights[2][2*d:3*d].reshape(d,1) 
bo = weights[2][3*d:4*d].reshape(d,1) 

Wy = weights[3].reshape(yd,d)                                                 
by = weights[4].reshape(yd,1) 
Uy = np.zeros((yd,ud))

##Need to change code!
#Uy = np.random.uniform(-1,1, size=(yd,ud))










'''
#Learned model with training inputs (true y_{t-1})
y_tr_vec = np.zeros((T,yd))
y_gen, c_gen, h_gen, v, zi, zf, zp, zo = generate(T,d, yd, 
                                         u.reshape(T,ud), 
                                         c_0.reshape(d), Sig_c.reshape(d), 
                                         h_0.reshape(d), Sig_h.reshape(d), 
                                         Sigma_y,
                                         Wy, Wi, Wf, Wp, Wo, 
                                         Uy, Ui, Uf, Up, Uo, 
                                         by.reshape(yd), bi.reshape(d), 
                                         bf.reshape(d), bp.reshape(d), 
                                         bo.reshape(d))
y_tr_vec += y_gen

print('u')
print(u[:10])
print('h')
print(h_gen[:10])
print('c')
print(c_gen[:10])
print('y')
print(y_gen[:10])
'''

#Learned model with training inputs (true y_{t-1})
y_tr_vec = np.zeros((T,yd))
h_gen = np.zeros((T,d))
c_gen= np.zeros((T,d))

for j in range(0,T):
        
    y_gen, c_0, h_0, v, zi, zf, zp, zo = generate(1,d, yd, 
                                         u[j,:,:].reshape(1,1), 
                                         c_0.reshape(d), Sig_c.reshape(d), 
                                         h_0.reshape(d), Sig_h.reshape(d), 
                                         Sigma_y,
                                         Wy, Wi, Wf, Wp, Wo, 
                                         Uy, Ui, Uf, Up, Uo, 
                                         by.reshape(yd), bi.reshape(d), 
                                         bf.reshape(d), bp.reshape(d), 
                                         bo.reshape(d))
    
    y_tr_vec[j,:] += y_gen.reshape(yd)
    h_gen[j,:] += h_0.reshape(d)
    c_gen[j,:] += c_0.reshape(d)





#Extrapolate, using generated y_{t-1} as input

y_tr_gen_vec = np.zeros((T_new,yd)) 
h_arr = np.zeros((T_new,d))
c_arr= np.zeros((T_new,d))

for j in range(0,T_new):
    if j ==0:
        y_tr_gen = y[-1,:,:].reshape(1,yd)
        
        c_0 = c_gen[-1,:]#Ec[-1,:,0].reshape(d) 
        h_0 = h_gen[-1,:]#Eh[-1,:,0].reshape(d)
        
    else: 
        y_tr_gen = y_tr_gen_vec[j-1,:].reshape(1,yd)
        c_0 = c_arr[j-1,:]
        h_0 = h_arr[j-1,:]
        

    y_tr_gen, c_0, h_0, _,_,_,_,_ = generate(1,d, yd, y_tr_gen, 
                                         c_0.reshape(d), Sig_c.reshape(d), 
                                         h_0.reshape(d), Sig_h.reshape(d), 
                                         Sigma_y,
                                         Wy, Wi, Wf, Wp, Wo, 
                                         Uy, Ui, Uf, Up, Uo, 
                                         by.reshape(yd), bi.reshape(d), 
                                         bf.reshape(d), bp.reshape(d), 
                                         bo.reshape(d))
    y_tr_gen_vec[j,:] += y_tr_gen.reshape(yd)
    c_arr[j,:] = c_0.reshape(d)
    h_arr[j,:] = h_0.reshape(d)
    
        

timestamp = time.time()
path = 'images/{}'.format(timestamp)
os.mkdir(path)

     



plt.plot(t[1:],y_full.reshape(T_full))
plt.plot(t[1:stop],y_tr_vec.reshape(T))
#plt.plot(t[stop:],y_test_vec.reshape(T_test))
#plt.plot(t[stop:],y_test_vec2.reshape(T_test),'m')
#plt.plot(t[1:],y.reshape(T_full))
#plt.plot(t[1:],y_gen_vec.reshape(T_full))
#plt.plot(t[1:],y_gen.reshape(T))
#r1 = np.arange(t[1],end+T_new*dt, dt)
#plt.plot(t[1:], y_tr_gen_vec)
#plt.plot(r1, y_tr_gen_vec)

'''
r = np.arange(t[-1]+dt, t[-1]+dt+T_new*dt, dt)
plt.plot(r, y_tr_gen_vec)
'''

#r = np.arange(t[stop]+dt, t[stop]+dt+T_new*dt, dt)
r = np.arange(stop, stop+T_new,dt)
#plt.plot(r,y_test_vec2.reshape(T_new),'m')
plt.plot(r, y_tr_gen_vec, 'r')


plt.savefig(path+ '/d{}_N{}_varh{}_varc{}.png'.format( d, N,  var_h,  var_c))

plt.show()



        
y_tr_gen= np.array([.06595209]).reshape(1,1)
c_0 = np.array([-.03640367, -.02395167, -.07682351])
h_0 = np.array([-.02159981, -.01323847, -.04119405])

y_tr_gen, c_0, h_0, _,_,_,_,_ = generate(1,d, yd, y_tr_gen, 
                                         c_0.reshape(d), Sig_c.reshape(d), 
                                         h_0.reshape(d), Sig_h.reshape(d), 
                                         Sigma_y,
                                         Wy, Wi, Wf, Wp, Wo, 
                                         Uy, Ui, Uf, Up, Uo, 
                                         by.reshape(yd), bi.reshape(d), 
                                         bf.reshape(d), bp.reshape(d), 
                                         bo.reshape(d))

print('y')
print(y_tr_gen)
print('h')
print(h_0)
print('c')
print(c_0)
