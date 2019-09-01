import numpy as np
from scipy.special import expit
import sys
import matplotlib.pyplot as plt
from E_PG import qdf_log_pdf, entropy_q, qdf_log_pdf_vec, entropy_q_vec
from scipy import integrate

def generate(T, mu_0, covar_c, h_0, covar_h, 
             Wi, Wf, Wp, Wo, Ui, Uf, Up, Uo, bi, bf, bp, bo, u):
    c = np.zeros(T)
    h = np.zeros(T)
    v = np.zeros(T)
    zi = np.zeros(T)
    zf = np.zeros(T)
    zp = np.zeros(T)
    zo = np.zeros(T)

    zi[0] = np.random.binomial(1, expit(Wi*h_0+Ui*u[0]+bi))
    zf[0] = np.random.binomial(1, expit(Wf*h_0+Uf*u[0]+bf))
    zp[0] = np.random.binomial(1, expit(Wp*h_0+Up*u[0]+bp))
    zo[0] = np.random.binomial(1, expit(Wo*h_0+Uo*u[0]+bo))

    c[0] = np.random.normal(zf[0]*mu_0+zi[0]*(2*zp[0]-1),
                            np.sqrt(covar_c[0,0]))
    v[0] = np.random.binomial(1,  expit(c[0]))
    h[0] = np.random.normal(zo[0]*(2*v[0]-1), np.sqrt(covar_h[0,0])) 
    for t in range(1,T):
        zi[t] = np.random.binomial(1, expit(Wi*h[t-1]+Ui*u[t]+bi))
        zf[t] = np.random.binomial(1, expit(Wf*h[t-1]+Uf*u[t]+bf))
        zp[t] = np.random.binomial(1, expit(Wp*h[t-1]+Up*u[t]+bp))
        zo[t] = np.random.binomial(1, expit(Wo*h[t-1]+Uo*u[t]+bo))
        c[t] = np.random.normal(zf[t]*c[t-1]+zi[t]*(2*zp[t]-1),
                                np.sqrt(covar_c[t,t]))
        v[t] = np.random.binomial(1,  expit(c[t]))
        h[t] = np.random.normal(zo[t]*(2*v[t]-1), 
                                np.sqrt(covar_h[t,t])) 
    return c, h, v, zi, zf, zp, zo

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
    Emm = (Em[...,None]*Em[:,None,:]).reshape(Sigma.shape)+Sigma
    return Em, Emm, Sigma

def get_diags(Exx,d,T):
    '''Extracts diagonals and 1 off diagonals from dxTxT array '''
    diags = np.zeros((d,T))
    off_diags = np.zeros((d,T-1))
    for i in range(0,d):
        diags[i,:] = np.diag(Exx[i,:,:])
        off_diags[i,:] = np.diag(Exx[i,:,:], k=-1)
    return diags, off_diags

###############################################################

def Lambda_h_op(t, Eomega_star, W_star):
    value = W_star.T @ (Eomega_star[t+1,:,:][:,0]* W_star)
    return value


def Lambda_h_m_op( Ez_star, Eomega_star, W_star, 
                  U_star, b_star, u):
    value = W_star.T @ (Ez_star[1:,:,:]-1/2)
    value  += -W_star.T @ (Eomega_star[1:,:,:]
                           *(U_star @ u[1:,:,:]+b_star))
    return value


def update_qh(T,d,h_0, inv_covar, inv_covar_y, y, Wi, Wf, Wp, Wo, Wy, 
              u, Ui, Uf, Up, Uo, Uy, bi, bf, bp, bo, by,
              Ev, Eomega_i, Eomega_f, Eomega_p, Eomega_o, 
              Ezi, Ezf, Ezp, Ezo):
    
    #Construct Precision_d matrix
    #Lambda shape (T,d,d)
    Lambda = np.zeros((T,d,d))
    
    
    Lambda += Wy.T @ (inv_covar_y*Wy)
    Lambda += inv_covar*np.identity(d)
    for t in range(0,T-1):
        Lambda[t,:,:] += Lambda_h_op(t, Eomega_i, Wi)
        Lambda[t,:,:] += Lambda_h_op(t, Eomega_f, Wf)
        Lambda[t,:,:] += Lambda_h_op(t, Eomega_p, Wp)
        Lambda[t,:,:] += Lambda_h_op(t, Eomega_o, Wo)
    

    #Lambda_m shape = (T,d,1)
    Lambda_m = np.zeros((T,d,1))
    Lambda_m += (inv_covar_y*Wy).T @ (y-(Uy @ u + by))
    
    Lambda_m += inv_covar * ( Ezo*(2*Ev-1) )
    Lambda_m[:-1,:,:] += Lambda_h_m_op(Ezi, Eomega_i, Wi, Ui, bi, u)
    Lambda_m[:-1,:,:] += Lambda_h_m_op(Ezf, Eomega_f, Wf, Uf, bf, u)
    Lambda_m[:-1,:,:] += Lambda_h_m_op(Ezp, Eomega_p, Wp, Up, bp, u)
    Lambda_m[:-1,:,:] += Lambda_h_m_op(Ezo, Eomega_o, Wo, Uo, bo, u)

    
    return Lambda, Lambda_m
    
    
#############################################################
def update_q_gamma(Ecc_diags):
    g = np.sqrt(4*Ecc_diags)
    E_gamma = 1/(2*g)*np.tanh(g/2)
    return g, E_gamma


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

    g = np.sqrt(value)
    E_omega = 1/(2*g)*np.tanh(g/2)
    return g, E_omega
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
    
    value = W_Eh_z_update(Wp, Up, bp, u, Eh, h_0, value)
    
    return expit(value)

def update_zo(h_0, inv_covar, Wo, Uo, bo, u, Eh, Ev):
    value = inv_covar*Eh*(2*Ev-1) - 1/2*inv_covar
    value = W_Eh_z_update(Wo, Uo, bo, u, Eh, h_0, value)
    
    return expit(value)

                    

###########################################################
##ELBO calculation###


def elbo_c(T, d, c_0, inv_covar, Ec, Ecc_diags, Ecc_off_diags):
    value = np.zeros((T,d,1))
    value += np.log(np.sqrt(1/(2*np.pi)*inv_covar))

    value += -1/2*inv_covar*Ecc_diags

    value[0,:,:] += inv_covar*Ezf[0,:,:]*Ec[0,:,:]*c_0 
    value[1:,:,:] += inv_covar*Ezf[1:,:,:]*Ecc_off_diags

    value += inv_covar*Ezi*(2*Ezp-1)*Ec

    value[0,:,:] += -1/2*inv_covar*Ezf[0,:,:]*c_0**2
    value[1:,:,:] += -1/2*inv_covar*Ezf[1:,:,:]*Ecc_diags[:-1,:,:]

    value[0,:,:] += -inv_covar*Ezi[0,:,:]*Ezf[0,:,:]*(2*Ezp[0,:,:]
                                                      -1)*c_0
    value[1:,:,:] += -inv_covar*Ezi[1:,:,:]*Ezf[1:,:,:]*(2*Ezp[1:,:,:]
                                                      -1)*Ec[:-1,:,:]

    value += -1/2*inv_covar*Ezi
    
    print('elbo_c:', np.sum(value))
    return np.sum(value)

def elbo_h(T,d,h_0, inv_covar, Eh, Ehh_diags, Ezo, Ev):
    value = np.zeros((T,d,1))
    value += -1/2*inv_covar*Ehh_diags
    value += inv_covar*Ezo*(2*Ev-1)*Eh
    value += -1/2*inv_covar*Ezo
    
    
    log_term = T*(-d/2*np.log(2*np.pi)-1/2*np.sum(
            np.log((1/inv_covar)) ) )
    value = np.sum(value)+log_term
    
    print('elbo_h:', value)
    return value

def elbo_y(T, yd, y, h_0, inv_covar_y, u, Uy, by, Eh, Ehh):
    value = np.zeros((T,1))
    value += -1/2*np.sum(y*inv_covar_y*y,axis=1)
    
    Uu_plus_b = Uy @ u + by
    
    
    value[0,:] += np.sum(y[0,:,:]*(inv_covar_y*(Wy @ h_0 + Uu_plus_b[0,:,:]))) 
                                                
    value[1:,:] += np.sum(y[1:,:,:]*(inv_covar_y*(Wy @ Eh[:-1,:,:] + 
                                                Uu_plus_b[1:,:,:])),axis=1)

    
    value[0,:] += -1/2*np.trace(Wy.T @ (inv_covar_y*Wy) @ np.outer(h_0,h_0))
    value[1:,:] += -1/2*np.trace(Wy.T @ (inv_covar_y*Wy) @ Ehh[:-1,:,:], 
                             axis1=1, axis2 =2).reshape(T-1,1)

    value[0,:] += -np.sum( Uu_plus_b[0,:,:]*((inv_covar_y*Wy) @ h_0))
    value[1:,:] += -np.sum(Uu_plus_b[1:,:,:]*((inv_covar_y*Wy) @ Eh[:-1,:,:])
                           ,axis=1) 

    value += -1/2*np.sum(Uu_plus_b*(inv_covar_y*Uu_plus_b), axis=1)

    log_term = T*(-yd/2*np.log(2*np.pi)-1/2*np.sum(
            np.log((1/inv_covar_y)) ) )
    value = np.sum(value)+log_term
    
    print('elbo_y:', value)
    return value
    


def elbo_z_star(T,d,Ez,Eh, h_0, W_star, U_star, b_star, u, star):
    
    value = np.zeros((T,d,1))
    Uu_plus_b = U_star @ u + b_star

    value[0,:,:] += (Ez[0,:,:]-1/2)*(W_star @ h_0 + Uu_plus_b[0,:,:])

    value[1:,:,:] += (Ez[1:,:,:]-1/2)*(W_star @ Eh[:-1,:,:] 
                                       + Uu_plus_b[1:,:,:])
    value = np.sum(value)+T*d*np.log(1/2)
    print('elbo_{}'.format(star), value)
    return value


def elbo_v(T,d,Ev, Ec):
    value = 2*(Ev-1/2)*Ec
    value = np.sum(value)+T*d*np.log(1/2)
    print('elbo_v:', value)
    return value

def elbo_gamma1(T, d, E_gamma, Ecc_diags):
    value = -1/2*E_gamma*4*Ecc_diags
    print('elbo_gamma1:', np.sum( value) )
    return np.sum( value )

def elbo_omega_star_1(Eomega_star, g_star, star):
    value = -1/2*Eomega_star*(g_star)**2
    print('elbo_omega1_{}:'.format(star), np.sum( value) )
    return np.sum( value )

'''
def elbo_PG2(g, str):
    value = 0
    g_arr = g.ravel()
    for i in range(0,len(g_arr)):
        value += integrate.quad(qdf_log_pdf, 0, np.inf,
                           args=(1,0, g_arr[i]),
                           epsabs=1e-1, epsrel = 0)[0]
    print('elbo_{}_2'.format(str), value)
    return value
'''
def elbo_PG2_vec(g, str, L, h):
    num = int(L/h)
    a = .00001*np.ones(g.shape)
    b = L * np.ones(g.shape)
    x = np.linspace(a,b,num)
    y, qdf = qdf_log_pdf_vec(x,1,0,g)
    value = np.trapz(y,x,axis=0)
    value = np.sum(value)
    print('elbo_{}_2'.format(str), value)
    return value, qdf, x

    
def entropy_Gauss(T,d,Sigma,str):
    value = np.sum(1/2*np.log(np.linalg.det(Sigma)))
    value += ( (T*d)/2 )*(1+np.log(2*np.pi))
    print('Entropy_{}'.format(str), value)
    return value


def entropy_Bern(p, str):    
    value = -p*np.log(p)-(1-p)*np.log(1-p)
    value = np.nan_to_num(value, copy=False)
    print('entropy_{}:'.format(str), np.sum(value))
    return np.sum(value)

'''
def entropy_PG(g,str):
    value = 0
    g_arr = g.ravel()
    for i in range(0,len(g_arr)):
        value += integrate.quad(entropy_q, 0, np.inf,
                           args=(1, g_arr[i]),
                           epsabs=1e-1, epsrel = 0)[0]
    print('{}_entrpy'.format(str), value)
    return value
'''



def entropy_PG_vec(qdf,x,str):
    
    y = entropy_q_vec(qdf)
    value = np.trapz(y,x,axis=0)
    value = np.sum(value)
    print('entrpy_{}'.format(str), value)
    return value

def get_elbo(T, d, yd,  c_0, inv_covar_c, h_0, inv_covar_h, 
             Wi, Wf, Wp, Wo, y, inv_covar_y,
             Ui, Uf, Up, Uo, Uy, u,
             bi, bf, bp, bo, by,
             Ec, Ecc_diags, Ecc_off_diags, Sigma_c, 
             Ev, Eh, Ehh_diags, Ehh,Sigma_h, 
             Ezi, Ezf, Ezp, Ezo, E_gamma, g_gamma, 
             Eomega_i, Eomega_f, Eomega_p, Eomega_o, 
             gi, gf, gp , go, L, h):

    elbo = elbo_y(T, yd, y, h_0, inv_covar_y, u, Uy, by, Eh, Ehh)
    elbo += elbo_c(T, d, c_0, inv_covar_c, 
                  Ec, Ecc_diags, Ecc_off_diags)
    elbo += elbo_v(T, d, Ev, Ec)
    elbo += elbo_h(T, d, h_0, inv_covar_h, Eh, Ehh_diags, Ezo, Ev)

    elbo += elbo_z_star(T, d, Ezi, Eh, h_0, Wi, Ui, bi, u, 'zi')
    elbo += elbo_z_star(T, d, Ezf, Eh, h_0, Wf, Uf, bf, u, 'zf')
    elbo += elbo_z_star(T, d, Ezp, Eh, h_0, Wp, Up, bp, u, 'zp')
    elbo += elbo_z_star(T, d, Ezo, Eh, h_0, Wo, Uo, bo, u, 'zo')
    
    elbo += elbo_gamma1(T, d, E_gamma, Ecc_diags)
    elbo += elbo_omega_star_1(Eomega_i, gi, 'i')
    elbo += elbo_omega_star_1(Eomega_f, gf, 'f')
    elbo += elbo_omega_star_1(Eomega_p, gp, 'p')
    elbo += elbo_omega_star_1(Eomega_o, go, 'o')

    #integrate all PG variables together and save qdf for entropy calc
    PG = np.concatenate((g_gamma, gi, 
                         gf, gp, go), axis=0)
    value, qdf, x = elbo_PG2_vec(PG, 'Int_PG', L, h)

    elbo += value
    elbo += entropy_Gauss(T,d,Sigma_c,'c')

    elbo += entropy_Bern(Ev, 'v')
    elbo += entropy_Gauss(T,d,Sigma_h,'h')

    elbo += entropy_Bern(Ezi, 'zi')
    elbo += entropy_Bern(Ezf, 'zf')
    elbo += entropy_Bern(Ezp, 'zp')
    elbo += entropy_Bern(Ezo, 'zo')

    elbo += entropy_PG_vec(qdf, x, 'PG')
    return elbo

def get_diff(param, param_old, diff_list):
    param_diff = np.amax(np.absolute(param-param_old))
    diff_list.append(param_diff)
    param_old = param
    return diff_list, param_old

#####
seed = np.random.randint(0,10000)
#seed = 2782
np.random.seed(seed)
print('random_seed:',seed)

#elbo integration parameters
L = 10 #end of intergation interval
h = .01 #grid spacing

T=20
d = 10
ud = 5
yd = 1

var_c = .2
mu_c0 = 0
inv_covar_c = 1/var_c*np.ones((d,1))
c_0 = mu_c0*np.ones((d,1))

var_h = .3
mu_h0 = 0
inv_covar_h = 1/var_h*np.ones((d,1))
h_0 = mu_h0*np.ones((d,1))

u = np.random.uniform(-1,1, size=(T, ud, 1))

var_y = .4
inv_covar_y = 1/var_y*np.ones((yd,1))
y = np.random.uniform(-1,1, size = (T, yd,1))

print('y')
print(y)
print('u')
print(u)

'''
Wi = .1*np.ones((d,d)) 
Wf = .2*np.ones((d,d)) 
Wp = .3*np.ones((d,d)) 
Wo = -.4*np.ones((d,d))

Ui = .4*np.ones((d,ud))
Uf = .3*np.ones((d,ud)) 
Up = -.2*np.ones((d,ud)) 
Uo = .1*np.ones((d,ud)) 

bi = -.3*np.ones((d,1)) 
bf = .5*np.ones((d,1)) 
bp = .7*np.ones((d,1)) 
bo = .6*np.ones((d,1)) 
'''

Wi = np.random.uniform(-1,1, size=(d,d))
Wf = np.random.uniform(-1,1, size=(d,d))
Wp = np.random.uniform(-1,1, size=(d,d))
Wo = np.random.uniform(-1,1, size=(d,d))
Wy = np.random.uniform(-1,1, size=(yd,d))

Ui = np.random.uniform(-1,1, size=(d,ud))
Uf = np.random.uniform(-1,1, size=(d,ud))
Up = np.random.uniform(-1,1, size=(d,ud))
Uo = np.random.uniform(-1,1, size=(d,ud))
Uy = np.random.uniform(-1,1, size=(yd,ud))

bi = np.random.uniform(-1,1, size=(d,1))
bf = np.random.uniform(-1,1, size=(d,1))
bp = np.random.uniform(-1,1, size=(d,1))
bo = np.random.uniform(-1,1, size=(d,1))
by = np.random.uniform(-1,1, size=(yd,1))


'''     
#c,h,v,zi,zf,zp,zo = generate(T, mu_0, covar_c, h_0, 
 #                         covar_h, Wi, Wf, Wp, Wo, 
  #                         Ui, Uf, Up, Uo, bi, bf, bp, bo, u)

print('Mu_0=',mu_0)
print('covar_c=')
print(covar_c)
print('c:', c)
print('Sigmoid(c):', expit(c))
print('v:', v)
print('h_0:', h_0)
print('h:', h)
print('zi',zi)
print('zf', zf)
print('zp', zp)
print('zo', zo)
print(' ')
'''

#Initialize
E_gamma = .3*np.ones((T,d,1))
Ev = .5*np.ones((T,d,1))
Ezi = .5*np.ones((T,d,1))
Ezf = .5*np.ones((T,d,1))
Ezp = .5*np.ones((T,d,1))
Ezo = .5*np.ones((T,d,1))
Eomega_i = .5*np.ones((T,d,1))
Eomega_f = .5*np.ones((T,d,1))
Eomega_p = .5*np.ones((T,d,1))
Eomega_o = .5*np.ones((T,d,1))


Ec_old = np.ones((T,d,1))*np.inf
Ecc_old = np.ones((d,T,T))*np.inf
Eh_old = np.ones((T,d,1))*np.inf
Ehh_old = np.ones((T,d,d))*np.inf
Ev_old = np.ones((T,d,1))*np.inf
g_gamma_old = np.ones((T,d,1))*np.inf
Ezi_old = np.ones((T,d,1))*np.inf
Ezf_old = np.ones((T,d,1))*np.inf
Ezp_old = np.ones((T,d,1))*np.inf
Ezo_old = np.ones((T,d,1))*np.inf
gi_old = np.ones((T,d,1))*np.inf
gf_old = np.ones((T,d,1))*np.inf
gp_old = np.ones((T,d,1))*np.inf
go_old = np.ones((T,d,1))*np.inf



diff = np.inf
tol = .01


diff_vec = []
elbo_vec = []
k_vec = []
k = 0


while diff > tol:
    diff_list = []

    #update qc
    Lambda_c, Lambda_c_m  = update_qc(T,d, c_0, inv_covar_c, 
                               Ezi, Ezf, Ezp, Ev, E_gamma)

    Ec, Ecc, Sigma_c  = get_moments(Lambda_c, Lambda_c_m)
    Ec = to_dxT(T,d, Ec)
    Ecc_diags, Ecc_off_diags = get_diags(Ecc,d,T)
    Ecc_diags = np.reshape(Ecc_diags.ravel('F'),(T,d,1))
    Ecc_off_diags = np.reshape(Ecc_off_diags.ravel('F'), (T-1,d,1))

    diff_list, Ec_old = get_diff( Ec, Ec_old, diff_list)
    diff_list, Ecc_old = get_diff( Ecc, Ecc_old, diff_list)
    
    #update q_h:
    Lambda_h, Lambda_h_m = update_qh(T,d,h_0, inv_covar_h, inv_covar_y, y, 
                                     Wi, Wf, Wp, Wo, Wy, 
                                     u, Ui, Uf, Up, Uo, Uy, 
                                     bi, bf, bp, bo, by,
                                     Ev, Eomega_i, Eomega_f, 
                                     Eomega_p, Eomega_o, 
                                     Ezi, Ezf, Ezp, Ezo)

    Eh, Ehh, Sigma_h = get_moments(Lambda_h, Lambda_h_m)
    Ehh_diags,_ = get_diags(Ehh,T,d)
    Ehh_diags = np.reshape(Ehh_diags,(T,d,1))

    diff_list, Eh_old = get_diff( Eh, Eh_old, diff_list)
    diff_list, Ehh_old = get_diff( Ehh, Ehh_old, diff_list)

    #update qv
    Ev = update_qv(Eh, inv_covar_h, Ec, Ezo)
    diff_list, Ev_old = get_diff( Ev, Ev_old, diff_list)
        
    #update q_gamma
    g_gamma, E_gamma = update_q_gamma(Ecc_diags)
    diff_list, g_gamma_old = get_diff( g_gamma, g_gamma_old, 
                                       diff_list)
    

    #update q_omegas
    gi, Eomega_i = update_q_omega_star(T, d, Eh, Ehh, h_0, 
                                       Wi, Ui, bi, u)
    diff_list, gi_old = get_diff( gi, gi_old, diff_list)

    gf, Eomega_f = update_q_omega_star(T,d,Eh, Ehh, h_0, 
                                       Wf, Uf, bf, u)
    diff_list, gf_old = get_diff( gf, gf_old, diff_list)

    gp, Eomega_p = update_q_omega_star(T,d,Eh, Ehh, h_0, 
                                       Wp, Up, bp, u)
    diff_list, gp_old = get_diff(gp, gp_old, diff_list)

    go, Eomega_o = update_q_omega_star(T,d, Eh, Ehh, h_0, 
                                       Wo, Uo, bo, u)
    diff_list, go_old = get_diff( go, go_old, diff_list)
    
    #update q_zi
    Ezi = update_zi(T,d, c_0, inv_covar_c, Wi, Ui, bi, u, h_0, 
              Eh, Ec, Ezp, Ezf)
    diff_list, Ezi_old = get_diff( Ezi, Ezi_old, diff_list)

    #update q_zf
    Ezf = update_zf(T,d, c_0, inv_covar_c, Wf, Uf, bf, u, 
              h_0, Eh,  Ec, Ecc_diags, Ecc_off_diags, Ezi, Ezp)
    diff_list, Ezf_old = get_diff( Ezf, Ezf_old, diff_list)

    #update q_zp
    Ezp = update_zp(c_0, inv_covar_c, Wp, Up, bp, u, 
                h_0, Eh, Ec, Ezi, Ezf)
    diff_list, Ezp_old = get_diff( Ezp, Ezp_old, diff_list)
    
    #update q_zo
    Ezo = update_zo(h_0, inv_covar_h, Wo, Uo, bo, u, Eh, Ev)
    diff_list, Ezo_old = get_diff( Ezo, Ezo_old, diff_list)


    #convergence check
    diff = np.amax( diff_list )
    diff_vec.append(diff)
    
    if k % 1 ==0:
    
        elbo = get_elbo(T, d, yd, c_0, inv_covar_c, h_0, inv_covar_h, 
                        Wi, Wf, Wp, Wo, y, inv_covar_y,
                        Ui, Uf, Up, Uo, Uy, u,
                        bi, bf, bp, bo, by,
                        Ec, Ecc_diags, Ecc_off_diags, Sigma_c, 
                        Ev, Eh, Ehh_diags, Ehh, Sigma_h,
                        Ezi, Ezf, Ezp, Ezo, E_gamma, g_gamma, 
                        Eomega_i, Eomega_f, Eomega_p, Eomega_o, 
                        gi, gf, gp , go, L, h)
        elbo_vec.append(elbo)
        k_vec.append(k)
        print(' ')
    
    k+=1
    print(k)





print('Ec:')
print(Ec)

'''
print('Ec prior:')
Ec_mod = np.zeros(T)
Ec_mod[0] = mu_0
Ec_mod[1:] = Ec[:-1]
print(Ezf*Ec_mod+Ezi*(2*Ezp-1))
print(' ')
'''
print('Ecc:')
print(Ecc)
print('Sigma_c')
print(Sigma_c)
print(' ')
print('Eh:',Eh)

''' 
print('Eh prior:')
print(Ezo*(2*Ev-1))
'''

print(' ')
print('Ehh')
print(Ehh)

print('Sigma_h:')
print(Sigma_h)
print(' ')
print('Ev:')
print(Ev)
'''
print('Ev prior:')
print(expit(2*Ec))
'''
print(' ')
print('E_gamma:')
print(E_gamma)
print('gg:')
print(g_gamma)  
print(' ')
print('Ezi')
print(Ezi)

'''
Eh_mod = np.zeros(T)
Eh_mod[0] = h_0
Eh_mod[1:] = Eh[:-1]
print('Ezi prior:')
print(expit(Wi*Eh_mod+Ui*u+bi))
'''

print(' ')
print('Ezf')
print(Ezf)
'''
print('Ezf prior:')
print(expit(Wf*Eh_mod+Uf*u+bf))
'''
print(' ')

print('Ezp')
print(Ezp)
'''
print('Ezp prior:')
print(expit(Wp*Eh_mod+Up*u+bp))
'''
print(' ')
print('Ezo')
print(Ezo)
'''
print('Ezo prior:')
print(expit(Wo*Eh_mod+Uo*u+bo))
'''
print(' ')
print('E_omega_i:')
print(Eomega_i)
print('E_omega_f:')
print(Eomega_f)
print('E_omega_p:')
print(Eomega_p)
print('E_omega_o:')
print(Eomega_o)
print('gi')
print( gi)
print('gf')
print( gf)
print('gp')
print( gp)
print('go')
print( go)

print('elbo:', elbo)




plt.plot(k_vec, elbo_vec)
plt.xlabel('Iteration')
plt.ylabel('ELBO')
plt.title('ELBO convergence')
plt.savefig('ELBO_0.png')
plt.show()
plt.close()

plt.plot(np.arange(k), diff_vec)
plt.xlabel('Iteration')
plt.ylabel('Max Parameter Difference')
plt.savefig('Error.png')
plt.show()
plt.close()




