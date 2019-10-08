import numpy as np
from pypolyagamma import PyPolyaGamma
from scipy.special import expit
import sys
import time
import matplotlib.pyplot as plt
from PG_int import qdf_log_pdf, entropy_q, qdf_log_pdf_vec, entropy_q_vec
from scipy import integrate
import os



def generate(T,d, yd, u, c0, h0,
             Wy, Wi, Wf, Wp, Wo, 
             Ui, Uf, Up, Uo, 
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
    yg[0,:] = Wy @ hg[0,:]  + by
    for t in range(1,T):
        zi[t,:] = expit(Wi @ hg[t-1,:]+Ui @ u[t,:]+bi)  
        zf[t,:] = expit(Wf @ hg[t-1,:]+Uf @ u[t,:]+bf)  
        zp[t,:] = expit(2*(Wp @ hg[t-1,:]+Up @ u[t,:]+bp))
        zo[t,:] = expit(Wo @ hg[t-1,:]+Uo @ u[t,:]+bo) 
        cg[t,:] = zf[t,:]*cg[t-1,:]+zi[t,:]*(2*zp[t,:]-1) 

        v[t,:]  = expit(2*cg[t,:]) 
        hg[t,:] = zo[t,:]*(2*v[t,:]-1) 
        yg[t,:] = Wy @ hg[t,:]  + by 

    return yg, cg, hg, v, zi, zf, zp, zo



####update c ########################################################
def to_dxT(d,T, x):
    '''Takes array of shape (T,d,1) and converts to 
        shape (d,T,1)'''
    return np.reshape(x.ravel('F'), (d,T,1))

def update_qc(T,d, c_0, inv_covar_c, Ezi, Ezf, Ezp, Ev, E_gamma):
    inv_covar = np.reshape(inv_covar_c, (d,1,1))
    Lambda = np.zeros((d,T,T))
    Ezf_re = to_dxT(d,T, Ezf)
    E_gamma_re = to_dxT(d,T, E_gamma)
    #Construct Precision Diagonal
    diag = np.zeros((d,T,1))
    diag += inv_covar+4*E_gamma_re    
    diag[:,:-1,:] += inv_covar*Ezf_re[:,1:,:]
    Lambda +=  diag*np.identity(T)

    #Construct Precision Off Diagonals
    off_diag = -inv_covar*Ezf_re[:,1:,:]
    Lambda[:,:-1,1:] += off_diag*np.identity(T-1)
    Lambda[:,1:,:-1] += off_diag*np.identity(T-1)

    #Construct Precision times mean
    Lambda_m = np.zeros((d,T,1))
    Ezi_re = to_dxT(d,T, Ezi)
    Ezp_re = to_dxT(d,T, Ezp)
    Ev_re = to_dxT(d,T,Ev)
    mod_c0 = np.reshape(c_0, (d,1,1))


    Lambda_m[:,0,:] += (inv_covar*mod_c0*(
            Ezf_re[:,0,:].reshape(d,1,1) )).reshape(d,1)
    Lambda_m += inv_covar*Ezi_re*(2*Ezp_re-1)
    Lambda_m += 2*(Ev_re-.5)
    Lambda_m[:,:-1,:] += -inv_covar*Ezi_re[:,1:,:]*Ezf_re[:,1:,:]*(
        2*Ezp_re[:,1:,:]-1 )
    
    return  Lambda, Lambda_m

def get_moments(Lambda, Lambda_m):
    Sigma = np.linalg.inv(Lambda)
    Em = Sigma @ Lambda_m
    #Emm = (Em[...,None]*Em[:,None,:]).reshape(Sigma.shape)+Sigma
    return Em,Sigma

def get_diags(Exx,d,T):
    '''Extracts diagonals and 1 off diagonals from dxTxT array '''
    diags = np.zeros((d,T))
    off_diags = np.zeros((d,T-1))
    for i in range(0,d):
        diags[i,:] = np.diag(Exx[i,:,:])
        off_diags[i,:] = np.diag(Exx[i,:,:], k=-1)
    return diags, off_diags

###############################################################

def sample_h_post(T, d, Eh, Sigma_h):
    post_h = np.zeros((T,d))
    for t in range(0,T):
        post_h[t,:] = np.random.multivariate_normal(Eh[t,:,:].reshape(d), 
                                                    Sigma_h[t,:,:])
    
    return post_h.reshape(T,d,1)

def sample_c_post(T, d, Ec, Sigma_c):
    post_c = np.zeros((T,d))
    for j in range(0,d):
        post_c[:,j] = np.random.multivariate_normal(Ec[:,j,:].reshape(T), 
                                                    Sigma_c[j,:,:])
    return post_c.reshape(T,d,1)

'''
def sample_pg(g,T,d):
    sample = np.zeros((T,d,1))
    for t in range(0,T):
        for j in range(0,d):
            seed = np.random.randint(0,10000000)
            pg = PyPolyaGamma(seed)
            sample[t,j,:] = pg.pgdraw(1, g[t,j,:])
    return sample
'''

def sample_pg(g,T,d):
    sample = np.empty(T*d)
    pg = PyPolyaGamma(seed)
    g1 = g.ravel()
    pg.pgdrawv(np.ones(T*d), g1, sample)
    return sample.reshape(T,d,1)


###############################################################
def Lambda_h_op(t, d, Eomega_star, W_star):
    value = W_star.T @ (Eomega_star[t+1,:,:]* W_star)
    return value


def Lambda_h_m_op( Ez_star, Eomega_star, W_star, 
                  U_star, b_star, u):
    value = W_star.T @ (Ez_star[1:,:,:]-1/2)
    value  += -W_star.T @ (Eomega_star[1:,:,:]
                           *(U_star @ u[1:,:,:]+b_star))
    return value

def Lambda_h_mP_op( Ez_star, Eomega_star, W_star, 
                  U_star, b_star, u):
    value = 2*W_star.T @ (Ez_star[1:,:,:]-1/2)
    value  += -4*W_star.T @ (Eomega_star[1:,:,:]
                           *(U_star @ u[1:,:,:]+b_star))
    return value


def update_qh(T,d,h_0, inv_covar, inv_covar_y, y, Wi, Wf, Wp, Wo, Wy, 
              u, Ui, Uf, Up, Uo, bi, bf, bp, bo, by,
              Ev, Eomega_i, Eomega_f, Eomega_p, Eomega_o, 
              Ezi, Ezf, Ezp, Ezo):
    
    #Construct Precision_d matrix
    #Lambda shape (T,d,d)
    Lambda = np.zeros((T,d,d))
    
    Lambda += Wy.T @ inv_covar_y @ Wy
    Lambda += inv_covar*np.identity(d)
    
    for t in range(0,T-1):
        Lambda[t,:,:] += Lambda_h_op(t, d, Eomega_i, Wi)
        Lambda[t,:,:] += Lambda_h_op(t, d, Eomega_f, Wf)
        Lambda[t,:,:] += 4*Lambda_h_op(t, d, Eomega_p, Wp)
        Lambda[t,:,:] += Lambda_h_op(t, d, Eomega_o, Wo)

    #Lambda_m shape = (T,d,1)
    Lambda_m = np.zeros((T,d,1))
    Lambda_m += (inv_covar_y @ Wy).T @ (y-by)
    
    Lambda_m += inv_covar * ( Ezo*(2*Ev-1) )
    Lambda_m[:-1,:,:] += Lambda_h_m_op(Ezi, Eomega_i, Wi, Ui, bi, u)
    Lambda_m[:-1,:,:] += Lambda_h_m_op(Ezf, Eomega_f, Wf, Uf, bf, u)
    Lambda_m[:-1,:,:] += Lambda_h_mP_op(Ezp, Eomega_p, Wp, Up, bp, u)
    Lambda_m[:-1,:,:] += Lambda_h_m_op(Ezo, Eomega_o, Wo, Uo, bo, u)

    
    return Lambda, Lambda_m
    
    
#############################################################
def update_q_gamma(Ecc_diags):
    g = np.sqrt(4*Ecc_diags)
    return g


def update_q_omega_star(T,d,Eh, Ehh, h_0, W_star, U_star, b_star, u):
    value = np.zeros((T,d,1))
        
    for i in range(0,d):
        wwT = np.outer(W_star[i,:], W_star[i,:])
        tr_val = np.trace(wwT @ np.outer(h_0,h_0))
        value[0,i,:] += tr_val
        tr_val = np.trace(wwT @ Ehh[:-1,:,:], 
                          axis1=1, axis2=2).reshape(T-1,1)
        value[1:,i,:] += tr_val
    
    Uu_plus_b = U_star @ u + b_star
    
    value[0,:,:] += (2*W_star @ h_0)*Uu_plus_b[0,:,:]
    value[1:,:,:] += (2*W_star @ Eh[:-1,:,:])*Uu_plus_b[1:,:,:]
    
    value += Uu_plus_b**2
    
    check = value < np.zeros(value.shape)
    count= np.sum(check)
    if count >0:
        value[np.nonzero(check)]=0
    
    g = np.sqrt(value)
    return g

def update_q_omega_P(T,d,Eh, Ehh, h_0, W_star, U_star, b_star, u):
    value = np.zeros((T,d,1))
        
    for i in range(0,d):
        wwT = np.outer(W_star[i,:], W_star[i,:])
        tr_val = np.trace(wwT @ np.outer(h_0,h_0))
        value[0,i,:] += tr_val
        tr_val = np.trace(wwT @ Ehh[:-1,:,:], 
                          axis1=1, axis2=2).reshape(T-1,1)
        value[1:,i,:] += tr_val
    
    Uu_plus_b = U_star @ u + b_star
    
    value[0,:,:] += (2*W_star @ h_0)*Uu_plus_b[0,:,:]
    value[1:,:,:] += (2*W_star @ Eh[:-1,:,:])*Uu_plus_b[1:,:,:]
    
    value += Uu_plus_b**2
    
    check = value < np.zeros(value.shape)
    count= np.sum(check)
    if count >0:
        value[np.nonzero(check)]=0
    
    g = np.sqrt(4*value)
    return g

##############################################################

def update_qv(Eh, inv_covar, Ec, Ezo):
    value = 2*inv_covar*Eh*Ezo+2*Ec
    return expit(value)
############################################################

def W_Eh_z_update(W_star, U_star, b_star, u, Eh, h_0, value):
    value[0,:,:] += W_star @ h_0
    value[1:,:,:] += W_star @ Eh[:-1,:,:]
    value += U_star @ u + b_star
    return value

def update_zi(T,d, c_0, inv_covar, Wi, Ui, bi, u, h_0, 
              Eh, Ec, Ezp, Ezf):
    value = inv_covar*Ec*(2*Ezp-1)
    value[0,:,:] += -inv_covar*Ezf[0,:,:]*c_0*(2*Ezp[0,:,:]-1)
    value[1:,:,:] += -inv_covar*Ezf[1:,:,:]*Ec[:-1,:,:]*(
        2*Ezp[1:,:,:]-1 )
    value += -.5*inv_covar
    value = W_Eh_z_update(Wi, Ui, bi, u, Eh, h_0, value)

    return expit(value)

def update_zf(T,d, c_0, inv_covar, Wf, Uf, bf, u, 
              h_0, Eh,  Ec, Ecc_diags, Ecc_off_diags, Ezi, Ezp):
    value = np.zeros((T,d,1))
    value[0,:,:] += inv_covar*Ec[0,:,:]*c_0
    value[1:,:,:] += inv_covar*Ecc_off_diags
    
    value[0,:,:] += -1/2*inv_covar*c_0**2
    value[1:,:,:] += -1/2*inv_covar*Ecc_diags[:-1,:,:]
    
    value[0,:,:] += -inv_covar*c_0*Ezi[0,:,:]*(2*Ezp[0,:,:]-1)
    value[1:,:,:] += -inv_covar*Ec[:-1,:,:]*Ezi[1:,:,:]*(
        2*Ezp[1:,:,:]-1 )
    
    value = W_Eh_z_update(Wf, Uf, bf, u, Eh, h_0, value)
    
    return expit(value)

def update_zp(c_0, inv_covar, Wp, Up, bp, u, h_0, Eh, Ec, Ezi, Ezf):
    value = 2*inv_covar*Ec*Ezi

    value[0,:,:] += -2*inv_covar*Ezf[0,:,:]*c_0*Ezi[0,:,:]
    value[1:,:,:] += -2*inv_covar*Ezf[1:,:,:]*Ec[:-1,:,:]*Ezi[1:,:,:]
    
    value = W_Eh_z_update(2*Wp, 2*Up, 2*bp, u, Eh, h_0, value)
    
    return expit(value)

def update_zo(h_0, inv_covar, Wo, Uo, bo, u, Eh, Ev):
    value = inv_covar*Eh*(2*Ev-1) - 1/2*inv_covar
    value = W_Eh_z_update(Wo, Uo, bo, u, Eh, h_0, value)
    
    return expit(value)

                    

############################################################################
def get_diff(param, param_old, diff_list):
    param_diff = np.amax(np.absolute(param-param_old))
    diff_list.append(param_diff)
    param_old = param
    return diff_list, param_old
###########################################################################
#parameter updates

def get_Ex_Exx(T, d, ud, Eh, Ehh, u):
    ones = np.ones((T,1,1))
    
    #Construct Ex
    Ex = np.concatenate((Eh, u, ones), axis =1)
    
    #Construct Exx
    Exx = np.zeros((T,d+ud+1, d+ud+1))
    Exx[:,:d,:d] += Ehh
    
    EhuT = (Eh[...,None]*u[:,None,:]).reshape(T,d,ud)
    Exx[:,:d,d:-1] += EhuT
    Exx[:,d:-1,:d] += EhuT.transpose(0,2,1)
    
    uuT = (u[...,None]*u[:,None,:]).reshape(T,ud,ud)
    Exx[:,d:d+ud, d:d+ud] += uuT
    
    Exx[:,:d,-1] += Eh.reshape(T,d)
    Exx[:,-1,:d] += Eh.reshape(T,d)

    Exx[:,d:d+ud,-1] += u.reshape(T,ud)
    Exx[:,-1,d:d+ud] += u.reshape(T,ud)
    
    Exx[:,-1,-1] += 1
    return Ex, Exx

def get_Ex_Exx_No_U(T, d, Eh, Ehh):
    ones = np.ones((T,1,1))
    
    #Construct Ex
    Ex = np.concatenate((Eh, ones), axis =1)
    
    #Construct Exx
    Exx = np.zeros((T,d+1, d+1))
    Exx[:,:d,:d] += Ehh
    
    Exx[:,:d,-1] += Eh.reshape(T,d)
    Exx[:,-1,:d] += Eh.reshape(T,d)
    
    Exx[:,-1,-1] += 1
    return Ex, Exx

def update_W_bar_y(T, d, yd, y, Ex, Exx):

    sumExx = np.sum(Exx, axis = 0)

    #Construct yExT                                                          
    yExT  = (y[...,None]*Ex[:,None,:]).reshape(T,yd,d+1)
    sumyExT = np.sum(yExT, axis=0)

    return sumyExT @ np.linalg.inv(sumExx)



def update_Sigma_y(T, yd, y, W_bar, Ex, Exx):
    A = np.zeros((T, yd, yd))
    A += (y[...,None]*y[:,None,:]).reshape(T,yd,yd)
    WEx = W_bar @ Ex
    WExyT = (WEx[...,None]*y[:,None,:]).reshape(T,yd,yd)
    A += -WExyT-WExyT.transpose(0,2,1)
    A += W_bar @ Exx @ W_bar.T
    return (1/T)*np.sum(A, axis=0)

def get_Ehmin_Ehhmin(T, d, Eh, Ehh, h_0):
    #Construct Eh_min_1
    Eh_min = np.zeros((T,d,1))
    Eh_min[0,:,:] += h_0
    Eh_min[1:,:,:] += Eh[:-1,:,:]

    #Construct Eh_min_1_h_min_1.T
    Ehh_min = np.zeros((T,d,d))
    h0h0T = np.outer(h_0.reshape(d), h_0.reshape(d))
    Ehh_min[0,:,:] += h0h0T
    Ehh_min[1:,:,:] += Ehh[:-1,:,:]
    
    return Eh_min, Ehh_min

def update_W_bar_star(T, d, ud, Ex, Exx, Eomega_star, Ez_star):
    
    
    A = np.zeros((d, d+ud+1,d+ud+1))
    rhs = np.zeros((d,d+ud+1,1))
    for i in range(0,d):
        om_Exx = Eomega_star[:,i,:].reshape(T,1,1)*Exx
        
        A[i,:,:] = np.sum(om_Exx, axis = 0)
        
        EzEx = (Ez_star[:,i,:].reshape(T,1,1)-1/2)*Ex
        rhs[i,:,:] = np.sum(EzEx, axis=0)
    return np.linalg.solve(A,rhs).reshape(d,d+ud+1)


def extract_W_weights(W_bar, d, ud):
    W = W_bar[:,:d]
    U = W_bar[:,d:d+ud]
    b = W_bar[:,-1]
    return W, U, b

def extract_W_weights_No_U(W_bar, d):
    W = W_bar[:,:d]
    b = W_bar[:,-1]
    return W, b

def update_Sigma_H(T, d, Eh, Ehh_diags, Ezo, Ev):
    Ehh_u = Ehh_diags.reshape(T,d)
    Eh_u = Eh.reshape(T,d)
    Ezo_u = Ezo.reshape(T,d)
    Ev_u = Ev.reshape(T,d)
    A = Ehh_u-2*Eh_u*Ezo_u*(2*Ev_u-1)+Ezo_u
    Sig = 1/T*np.sum(A, axis=0)
    return Sig.reshape(d,1)

def update_Sigma_C(T,d, c_0, Ecc_diags, Ecc_off_diags, Ec, Ezi, Ezf, Ezp):
    c0 = c_0.reshape(1,d)
    Ec_u = Ec.reshape(T,d)
    Ecc_u = Ecc_diags.reshape(T,d)
    Ecc_min_u = np.concatenate((c0**2, Ecc_u[:-1,:]), axis=0)
    Ecc_off_u = Ecc_off_diags.reshape(T-1,d)
    Ecc_off_u = np.concatenate((Ec_u[0,:]*c0,Ecc_off_u), axis = 0)
    Ec_min_u = np.concatenate((c0, Ec_u[:-1,:]), axis=0)
    Ezi_u = Ezi.reshape(T,d)
    Ezf_u = Ezf.reshape(T,d)
    Ezp_u = Ezp.reshape(T,d)
    
    A = Ecc_u-2*Ezf_u*Ecc_off_u-2*Ec_u*Ezi_u*(2*Ezp_u-1)
    A += Ezf_u*Ecc_min_u+2*Ezf_u*Ec_min_u*Ezi_u*(2*Ezp_u-1)+Ezi_u
    Sig = 1/T*np.sum(A, axis=0)
    return Sig.reshape(d,1)



def LSTM(c0, h0, Wi, Wf, Wp, Wo, bi, bf, bp, bo, Wy, by):
    i = expit(Wi @ h0 + bi)
    f = expit(Wf @ h0 + bf)
    p = np.tanh(Wp @ h0 + bp)
    o = expit(Wo @ h0 + bo)

    c = f*c0 + i*p
    h = o*np.tanh(c)
    y = Wy @ h + by
    return y,c,h,i,f,p,o

#########################################################################
#####
seed = np.random.randint(0,10000)
#seed = 5776#2492 #9765#4687#  8604
np.random.seed(seed)
print('random_seed:',seed)


##### Data ##########################################################

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
u_full = np.zeros((T_full,1,1))#data[:-1].reshape(T_full,ud,1)

y_full = data[1:].reshape(T_full,ud,1)

stop = int(.8*end/dt)#int(3*len(u_full)/4)+1

T = stop-1
u = data[:stop-1].reshape(T,ud,1)#np.zeros((T,1,1))
y = data[1:stop].reshape(T,yd,1)

T_test = T_full-T
u_test = data[stop-1:-1].reshape(T_test,ud,1)#np.zeros((T,1,1))
y_test = data[stop:].reshape(T_test,yd,1)


#######################################################################


#Hyperparameters ##################################################3
d = 20     #dimension of h and c
var_c = np.random.uniform(.01, .3)#.3
var_h = np.random.uniform(.01, .3)#.3
N = 1     #Number of Monte Carlo samples
tol = .03 #convergence check



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
var_y = np.random.uniform(.001,.5) #.0001#np.random.uniform(.001,.5)
Sigma_y = var_y*np.identity(yd)
inv_covar_y = 1/var_y*np.identity(yd)

'''
#Random weights
Wi = np.random.uniform(-1,1, size=(d,d))
Wf = np.random.uniform(-1,1, size=(d,d))
Wp = np.random.uniform(-1,1, size=(d,d))
Wo = np.random.uniform(-1,1, size=(d,d))
Wy = np.random.uniform(-1,1, size=(yd,d))

#Wy = np.zeros((1,d))
#Wy[0,0] = 1


Ui = np.random.uniform(-1,1, size=(d,ud))
Uf = np.random.uniform(-1,1, size=(d,ud))
Up = np.random.uniform(-1,1, size=(d,ud))
Uo = np.random.uniform(-1,1, size=(d,ud))

#Ui = np.zeros((d,1))
#Uf = np.zeros((d,1))
#Up = np.zeros((d,1))
#Uo = np.zeros((d,1))


bi = np.random.uniform(-1,1, size=(d,1))
bf = np.random.uniform(-1,1, size=(d,1))
bp = np.random.uniform(-1,1, size=(d,1))
bo = np.random.uniform(-1,1, size=(d,1))
by = np.random.uniform(-1,1, size=(yd,1))
#by = np.zeros((yd,1))
'''


#Loaded weights
wfile = 'weights/'+'LSTM_d20_eps100_lr0.0001_end200_1569288491.5787299.npy'
weights = np.load(wfile)

Ui = weights[0][0,:d].reshape(d,ud)   
Uf = weights[0][0,d:2*d].reshape(d,ud)       
Up = weights[0][0,2*d:3*d].reshape(d,ud) 
Uo = weights[0][0,3*d:4*d].reshape(d,ud) 

Wi = weights[1][:,:d].T 
Wf = weights[1][:,d:2*d].T                                                    
Wp = weights[1][:,2*d:3*d].T
Wo = weights[1][:,3*d:4*d].T
                                                                               
bi = weights[2][:d].reshape(d,1)
bf = weights[2][d:2*d].reshape(d,1)
bp = weights[2][2*d:3*d].reshape(d,1) 
bo = weights[2][3*d:4*d].reshape(d,1) 

Wy = weights[3].reshape(yd,d)                                                 
by = weights[4].reshape(yd,1) 



#Generate priors for training data
y_tr_vec = np.zeros((T,yd))
y_gen, Ec, Eh, Ev, Ezi, Ezf, Ezp, Ezo = generate(T,d, yd, 
                                         u.reshape(T,ud), 
                                         c_0.reshape(d), 
                                         h_0.reshape(d), 
                                         Wy, Wi, Wf, Wp, Wo, 
                                         Ui, Uf, Up, Uo, 
                                         by.reshape(yd), bi.reshape(d), 
                                         bf.reshape(d), bp.reshape(d), 
                                         bo.reshape(d))
y_tr_vec += y_gen




#Initialize
#Eh = np.random.uniform(-1,1,size=((T,d,1)))
h =  Eh.reshape(T,d,1)
hh = (h[...,None]*h[:,None,:]).reshape(T,d,d)
hh_diags = h**2


#Ec = np.random.uniform(-1,1,size=((T,d,1)))
c = Ec.reshape(T,d,1)
cc = c**2
cc_off = c[1:,:,:]*c[:-1,:,:]

Ev = Ev.reshape(T,d,1) 
v = np.random.binomial(1, Ev, size=(T,d,1))

Ezi = Ezi.reshape(T,d,1) 
zi = np.random.binomial(1, Ezi, size=(T,d,1))

Ezf = Ezf.reshape(T,d,1) 
zf = np.random.binomial(1, Ezf, size=(T,d,1))

Ezp = Ezp.reshape(T,d,1) 
zp = np.random.binomial(1, Ezp, size=(T,d,1))

Ezo = Ezo.reshape(T,d,1) 
zo = np.random.binomial(1, Ezo, size=(T,d,1))


Sig_h = update_Sigma_H(T, d, h, hh_diags, zo, v)
inv_covar_h = 1/Sig_h

Sig_c = update_Sigma_C(T,d, c_0, cc, cc_off, c, zi, zf, zp)
inv_covar_c = 1/Sig_c
'''


W_bar_y = np.concatenate((Wy, by), axis =1)
x_y, xx_y = get_Ex_Exx_No_U(T, d, h, hh) 
Sigma_y = update_Sigma_y(T, yd, y, W_bar_y, x_y, xx_y)
inv_covar_y = np.linalg.inv(Sigma_y)
'''


W_bar_y_old = np.ones((yd, d+1))*np.inf
Sigma_y_old = np.ones((yd, yd))*np.inf
W_bar_i_old = np.ones((d, d+ud+1))*np.inf
W_bar_f_old = np.ones((d, d+ud+1))*np.inf
W_bar_p_old = np.ones((d, d+ud+1))*np.inf
W_bar_o_old = np.ones((d, d+ud+1))*np.inf
Sig_h_old = np.ones((d,1))*np.inf
Sig_c_old = np.ones((d,1))*np.inf


diff = np.inf
diff_vec = []
elbo_vec = []
k_vec = []
k = 0


timestamp = time.time()
path = 'images/{}'.format(timestamp)


os.mkdir(path)



#while diff > tol:
for k in range(0,100):
    diff_list = []
        
    #update q_gamma
    g_gamma = update_q_gamma(cc)    
    gamma = sample_pg(g_gamma,T,d)
    
    #update q_omegas
    gi = update_q_omega_star(T, d, h, hh, h_0, Wi, Ui, bi, u)
    omega_i = sample_pg(gi,T,d)


    gf = update_q_omega_star(T,d, h, hh, h_0, Wf, Uf, bf, u)
    omega_f = sample_pg(gf,T,d)

    gp = update_q_omega_P(T,d, h, hh, h_0, Wp, Up, bp, u)
    omega_p = sample_pg(gp,T,d)
        

    go = update_q_omega_star(T,d, h, hh, h_0, Wo, Uo, bo, u)
    omega_o = sample_pg(go,T,d)
        

    #update q_zi
    Ezi = update_zi(T,d, c_0, inv_covar_c, Wi, Ui, bi, u, h_0, 
                        h, c, zp, zf)
    zi = np.random.binomial(1, Ezi, size=(T,d,1))
    #update q_zf
    Ezf = update_zf(T,d, c_0, inv_covar_c, Wf, Uf, bf, u, 
                        h_0, h, c, cc, cc_off, zi, zp)
    zf = np.random.binomial(1, Ezf, size=(T,d,1))

    #update q_zp
    Ezp = update_zp(c_0, inv_covar_c, Wp, Up, bp, u, 
                    h_0, h, c, zi, zf)
    zp = np.random.binomial(1, Ezp, size=(T,d,1))
        
        
    #update q_zo
    Ezo = update_zo(h_0, inv_covar_h, Wo, Uo, bo, u, h, v)
    zo = np.random.binomial(1, Ezo, size=(T,d,1))

    #update qc
    Lambda_c, Lambda_c_m  = update_qc(T,d, c_0, inv_covar_c, 
                                          zi, zf, zp, v, gamma)

    Ec, Sigma_c  = get_moments(Lambda_c, Lambda_c_m)
    Ec = to_dxT(T,d, Ec)
        

    c = sample_c_post(T, d, Ec, Sigma_c)
    cc = c**2
    cc_off = c[1:,:,:]*c[:-1,:,:]

    
    #update qv
    Ev = update_qv(h, inv_covar_h, c, zo)
    v = np.random.binomial(1, Ev, size=(T,d,1))
    
    #update q_h:
    Lambda_h, Lambda_h_m = update_qh(T,d,h_0, inv_covar_h, inv_covar_y, y, 
                                     Wi, Wf, Wp, Wo, Wy, 
                                     u, Ui, Uf, Up, Uo, 
                                     bi, bf, bp, bo, by,
                                     v, omega_i, omega_f, 
                                     omega_p, omega_o, 
                                     zi, zf, zp, zo)

    Eh, Sigma_h = get_moments(Lambda_h, Lambda_h_m)
    h = sample_h_post(T, d, Eh, Sigma_h)
    hh = (h[...,None]*h[:,None,:]).reshape(T,d,d)
    hh_diags = h**2
        
    '''
    ##### Update weights ####
    h_min, hh_min = get_Ehmin_Ehhmin(T, d, h, hh, h_0)
    x, xx = get_Ex_Exx(T, d, ud, h, hh, u)

        
    #Update Wi                                                               
    W_bar_i = update_W_bar_star(T, d, ud, x, xx, omega_i, zi)
    diff_list, W_bar_i_old = get_diff( W_bar_i, W_bar_i_old, diff_list)
    #Wi, bi = extract_W_weights_No_U(W_bar_i,  d)                            
    Wi, Ui, bi = extract_W_weights(W_bar_i, d, ud)
    bi = bi.reshape(d,1)


    #Update Wf                                                               
    W_bar_f = update_W_bar_star(T, d, ud, x, xx, omega_f, zf)
    diff_list, W_bar_f_old = get_diff( W_bar_f, W_bar_f_old, diff_list)
    #Wf, bf = extract_W_weights_No_U(W_bar_f,  d)                            
    Wf, Uf, bf = extract_W_weights(W_bar_f, d, ud)
    bf = bf.reshape(d,1)


    #Update Wp                                                               
    W_bar_p = update_W_bar_star(T, d, ud, x, xx, 2*omega_p, zp)
    diff_list, W_bar_p_old = get_diff( W_bar_p, W_bar_p_old, diff_list)
    #Wp, bp = extract_W_weights_No_U(W_bar_p,  d)                            
    Wp, Up, bp = extract_W_weights(W_bar_p,  d, ud)
    bp = bp.reshape(d,1)


    #Update Wo                                                               
    W_bar_o = update_W_bar_star(T, d, ud, x, xx, omega_o, zo)
    diff_list, W_bar_o_old = get_diff( W_bar_o, W_bar_o_old, diff_list)
    #Wo,  bo = extract_W_weights_No_U(W_bar_o,  d)                           
    Wo, Uo, bo = extract_W_weights(W_bar_o, d, ud)
    bo = bo.reshape(d,1)
    '''
    
    #Update Sig_h                                                            
    Sig_h = update_Sigma_H(T, d, h, hh_diags, zo, v)
    diff_list, Sig_h_old = get_diff( Sig_h, Sig_h_old, diff_list)
    inv_covar_h = 1/Sig_h

    #Update Sig_c                                                            
    Sig_c = update_Sigma_C(T,d, c_0, cc, cc_off, c, zi, zf, zp)
    diff_list, Sig_c_old = get_diff( Sig_c, Sig_c_old, diff_list)
    inv_covar_c = 1/Sig_c

    '''
    #Update Wy                                                               
    x_y, xx_y = get_Ex_Exx_No_U(T, d, h, hh)
    W_bar_y = update_W_bar_y(T, d, yd, y, x_y, xx_y)
    diff_list, W_bar_y_old = get_diff( W_bar_y, W_bar_y_old, diff_list)
    Wy, by = extract_W_weights_No_U(W_bar_y,  d)
    by = by.reshape(yd,1)


    #Update Sigma_y                                                          
    Sigma_y = update_Sigma_y(T, yd, y, W_bar_y, x_y, xx_y)
    diff_list, Sigma_y_old = get_diff( Sigma_y, Sigma_y_old, diff_list)
    inv_covar_y = np.linalg.inv(Sigma_y)
    '''

    
    ########################
    
    #convergence check
    diff = np.amax( diff_list )
    diff_vec.append(diff)
    print('diff:', diff)
    print('argmax_diff:',np.argmax(diff_list))
    
    
    if k % 10 == 0:
        y_tr_vec = np.zeros((T,yd))
        y_gen, c_gen, h_gen, _, _, _, _, _ = generate(T,d, yd, 
                                         u.reshape(T,ud), 
                                         c_0.reshape(d), 
                                         h_0.reshape(d), 
                                         Wy, Wi, Wf, Wp, Wo, 
                                         Ui, Uf, Up, Uo, 
                                         by.reshape(yd), bi.reshape(d), 
                                         bf.reshape(d), bp.reshape(d), 
                                         bo.reshape(d))
        y_tr_vec += y_gen
        plt.plot(t[1:stop],y_tr_vec.reshape(T))

        
        #Extrapolate, using generated y_{t-1} as input

        y_tr_gen_vec = np.zeros((T_new,yd)) 
        h_arr = np.zeros((T_new,d))
        c_arr= np.zeros((T_new,d))
        
        for j in range(0,T_new):
            if j ==0:
                y_tr_gen = y[-1,:,:].reshape(1,yd)
        
                c0 = Ec[-1,:,0].reshape(d) 
                h0 = Eh[-1,:,0].reshape(d)
        
            else: 
                y_tr_gen = y_tr_gen_vec[j-1,:].reshape(1,yd)
                c0 = c_arr[j-1,:]
                h0 = h_arr[j-1,:]
        

            y_tr_gen, c0, h0, _,_,_,_,_ = generate(1,d, yd, y_tr_gen, 
                                         c0.reshape(d),  
                                         h0.reshape(d), 
                                         Wy, Wi, Wf, Wp, Wo, 
                                         Ui, Uf, Up, Uo, 
                                         by.reshape(yd), bi.reshape(d), 
                                         bf.reshape(d), bp.reshape(d), 
                                         bo.reshape(d))
            y_tr_gen_vec[j,:] += y_tr_gen.reshape(yd)
            c_arr[j,:] = c0.reshape(d)
            h_arr[j,:] = h0.reshape(d)


        r = np.arange(stop, stop+T_new,dt)
        plt.plot(r, y_tr_gen_vec, label = '{}'.format(diff))
        plt.savefig(path+ '/d{}__varh{}_varc{}.png'.format(  d, var_h,  var_c))
       

    
    #if k%1 == 0:
    #    f1 = (Wi @ Eh[:-1,:])[:,0].reshape(T-1)
    #    f2 = bi[0].reshape(1)*np.ones(T-1)
    #    plt.plot(t[2:stop], f2, label=k)
        #plt.plot(t[1:stop], Eomega_o[:,0].reshape(T), label=k)
        #plt.plot(t[1:stop],Ezi[:,0].reshape(T), label=k)
     #   plt.plot(t[2:stop], Ezf[1:,0].reshape(T-1)*Ec[:-1,0].reshape(T-1)+
      #           Ezi[1:,0].reshape(T-1)*(2*Ezp[1:,0].reshape(T-1)-1))
        #plt.plot(t[1:stop], Eh[:,0,:].reshape(T), label = str(k))
        #plt.plot(t[1:stop], Ezo[:,0,:].reshape(T)*(2*Ev[:,0,:].reshape(T)-1)) 

        #plt.plot(t[1:stop], Ezo[:,0,:].reshape(T), label='{}_Ezo'.format(k))
        #plt.plot(t[1:stop], (2*Ev[:,0,:].reshape(T)-1), 
        #label='{}_2*Ev-1'.format(k))
    
    k+=1
    print('iteration:',k)
    print(' ')


plt.close()

'''
print('Ec:')
print(np.round(Ec,4))




print(' ')
print('Eh:',np.round(Eh,4))


print(' ')
print('Ev:')
print(np.round(Ev,4))


print(' ')
print('Ezi')
print(np.round(Ezi,4))


print(' ')
print('Ezf')
print(np.round(Ezf,4))

print(' ')

print('Ezp')
print(np.round(Ezp,4))

print(' ')
print('Ezo')
print(np.round(Ezo,4))

print('Sigma_y')
print(Sigma_y)
'''

'''
def sample_post(T, d, Eh, Sigma_h, Ec, Sigma_c):
    post_h = np.zeros((T,d))
    post_c = np.zeros((T,d))
    for t in range(0,T):
        post_h[t,:] = np.random.multivariate_normal(Eh[t,:,:].reshape(d), 
                                                    Sigma_h[t,:,:])
    for j in range(0,d):
        post_c[:,j] = np.random.multivariate_normal(Ec[:,j,:].reshape(T), 
                                                    Sigma_c[j,:,:])
    return post_h, post_c


def sample_last_post(N, d, Eh, Sigma_h, Ec, Sigma_c):
    post_h = np.zeros((N,d))
    post_c = np.zeros((N,d))
    for n in range(0,N):
        post_h[n,:] = np.random.multivariate_normal(Eh[-1,:,:].reshape(d), 
                                                    Sigma_h[-1,:,:])
        
        post_c[n,:] = np.random.normal(Ec[-1,:,:].reshape(d), 
                                       np.sqrt(Sigma_c[:,-1,-1]))
    return post_h, post_c
    
'''



'''
print('Sampling from posterior..') 
hpost, cpost = sample_last_post(N, d, Eh, Sigma_h, Ec, Sigma_c)
'''

'''
#Learned model with training inputs (true y_{t-1})
y_tr_vec = np.zeros((T,yd))
for n in range(0,N):
    y_gen, c_gen, h_gen, v, zi, zf, zp, zo = generate_sample(T,d, yd, 
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
y_tr_vec = 1/N*y_tr_vec



#Learned model on new data but using true previous y as input
y_test_vec = np.zeros((T_test,yd))
for n in range(0,N):
    y_gen, c_gen, h_gen, v, zi, zf, zp, zo = generate_sample(T_test,d, yd, 
                                         u_test.reshape(T_test,ud), 
                                         c_0.reshape(d), Sig_c.reshape(d), 
                                         h_0.reshape(d), Sig_h.reshape(d), 
                                         Sigma_y,
                                         Wy, Wi, Wf, Wp, Wo, 
                                         Uy, Ui, Uf, Up, Uo, 
                                         by.reshape(yd), bi.reshape(d), 
                                         bf.reshape(d), bp.reshape(d), 
                                         bo.reshape(d))
    y_test_vec += y_gen
y_test_vec = 1/N*y_test_vec




#Extrapolate, using generated y_{t-1} as input
    
y_tr_gen_vec = np.zeros((T_new,yd)) 
h_arr = np.zeros((T_new,N,d))
c_arr= np.zeros((T_new,N,d))

for j in range(0,T_new):
    y_tr_gen_sum = np.zeros(yd)
    for n in range(0,N):
        if j ==0:
            y_tr_gen = y[-1,:,:].reshape(1,yd)
            c_0 = cpost[n,:]
            h_0 = hpost[n,:]
        
        else: 
            y_tr_gen = y_tr_gen_vec[j-1,:].reshape(1,yd)
            c_0 = c_arr[j-1,n,:]
            h_0 = h_arr[j-1,n,:]
    
        y_tr_gen, c_0, h_0, _,_,_,_,_ = generate_sample(1,d, yd, y_tr_gen, 
                                         c_0.reshape(d), Sig_c.reshape(d), 
                                         h_0.reshape(d), Sig_h.reshape(d), 
                                         Sigma_y,
                                         Wy, Wi, Wf, Wp, Wo, 
                                         Uy, Ui, Uf, Up, Uo, 
                                         by.reshape(yd), bi.reshape(d), 
                                         bf.reshape(d), bp.reshape(d), 
                                         bo.reshape(d))
        y_tr_gen_sum += y_tr_gen.reshape(yd)
        c_arr[j,n,:] = c_0.reshape(d)
        h_arr[j,n,:] = h_0.reshape(d)
        
        
    y_tr_gen_vec[j,:] = 1/N*y_tr_gen_sum
'''


#Learned model with training inputs (true y_{t-1})
y_tr_vec = np.zeros((T,yd))
y_gen, c_gen, h_gen, v, zi, zf, zp, zo = generate(T,d, yd, 
                                         u.reshape(T,ud), 
                                         c_0.reshape(d),  
                                         h_0.reshape(d), 
                                         Wy, Wi, Wf, Wp, Wo, 
                                         Ui, Uf, Up, Uo, 
                                         by.reshape(yd), bi.reshape(d), 
                                         bf.reshape(d), bp.reshape(d), 
                                         bo.reshape(d))
y_tr_vec += y_gen

'''
#Learned model on new data but using true previous y as input
y_test_vec = np.zeros((T_test,yd))

y_gen, c_gen, h_gen, v, zi, zf, zp, zo = generate(T_test,d, yd, 
                                         u_test.reshape(T_test,ud), 
                                         c_0.reshape(d), Sig_c.reshape(d), 
                                         h_0.reshape(d), Sig_h.reshape(d), 
                                         Sigma_y,
                                         Wy, Wi, Wf, Wp, Wo, 
                                         Ui, Uf, Up, Uo, 
                                         by.reshape(yd), bi.reshape(d), 
                                         bf.reshape(d), bp.reshape(d), 
                                         bo.reshape(d))
y_test_vec += y_gen
'''

'''
##Test code below using learned model on new data with previous y as 
#input, but one at a time
y_test_vec2 = np.zeros((T_test,yd))
h_arr = np.zeros((T_test,d))
c_arr= np.zeros((T_test,d))

for j in range(0,T_test):
    if j == 0:
        y_gen = u_test[j,:,:].reshape(1,ud)
        c_0 = Ec[-1,:,0].reshape(d) 
        h_0 = Eh[-1,:,0].reshape(d)
    else:
        y_gen = u_test[j,:,:].reshape(1,ud)
        c_0 = c_arr[j-1,:]
        h_0 = h_arr[j-1,:]

    

    
    y_gen, c_0, h_0, _, _, _, _, _ = generate(1,d, yd, 
                                         y_gen.reshape(1,yd), 
                                         c_0.reshape(d), Sig_c.reshape(d), 
                                         h_0.reshape(d), Sig_h.reshape(d), 
                                         Sigma_y,
                                         Wy, Wi, Wf, Wp, Wo, 
                                         Ui, Uf, Up, Uo, 
                                         by.reshape(yd), bi.reshape(d), 
                                         bf.reshape(d), bp.reshape(d), 
                                         bo.reshape(d))
    y_test_vec2[j,:] += y_gen.reshape(yd)
    c_arr[j,:] = c_0.reshape(d)
    h_arr[j,:] = h_0.reshape(d)
    
'''


#Extrapolate, using generated y_{t-1} as input

y_tr_gen_vec = np.zeros((T_new,yd)) 
h_arr = np.zeros((T_new,d))
c_arr= np.zeros((T_new,d))

for j in range(0,T_new):
    if j ==0:
        #y_tr_gen = y[-1,:,:].reshape(1,yd)
        input = y[-1,:,:].reshape(1,yd)#np.zeros(1).reshape(1,1)
        ct =  Ec[-1,:,0].reshape(d) 
        ht = Eh[-1,:,0].reshape(d)
        
    else: 
        #y_tr_gen = y_tr_gen_vec[j-1,:]
        input = y_tr_gen_vec[j-1,:].reshape(1,yd)#np.zeros(1).reshape(1,1)
        ct = c_arr[j-1,:]
        ht = h_arr[j-1,:]
        

    y_tr_gen, ct, ht, _,_,_,_,_ = generate(1,d, yd, input, 
                                         ct.reshape(d),  
                                         ht.reshape(d),
                                         Wy, Wi, Wf, Wp, Wo, 
                                         Ui, Uf, Up, Uo, 
                                         by.reshape(yd), bi.reshape(d), 
                                         bf.reshape(d), bp.reshape(d), 
                                         bo.reshape(d))
    y_tr_gen_vec[j,:] += y_tr_gen.reshape(yd)
    c_arr[j,:] = ct.reshape(d)
    h_arr[j,:] = ht.reshape(d)
    
        
     



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


