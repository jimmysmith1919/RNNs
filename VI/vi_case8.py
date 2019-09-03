import numpy as np
from scipy.special import expit
import sys
import matplotlib.pyplot as plt
from PG_int import qdf_log_pdf, entropy_q, qdf_log_pdf_vec, entropy_q_vec
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

def Lambda_h_op(t, d, Eomega_star, W_star):
    value = W_star.T @ (Eomega_star[t+1,:,:]* W_star)
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
    
    Lambda += Wy.T @ inv_covar_y @ Wy
    Lambda += inv_covar*np.identity(d)
    
    for t in range(0,T-1):
        Lambda[t,:,:] += Lambda_h_op(t, d, Eomega_i, Wi)
        Lambda[t,:,:] += Lambda_h_op(t, d, Eomega_f, Wf)
        Lambda[t,:,:] += Lambda_h_op(t, d, Eomega_p, Wp)
        Lambda[t,:,:] += Lambda_h_op(t, d, Eomega_o, Wo)

    #Lambda_m shape = (T,d,1)
    Lambda_m = np.zeros((T,d,1))
    Lambda_m += (inv_covar_y @ Wy).T @ (y-(Uy @ u + by))
    
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


def elbo_y(T, yd, y, inv_covar_y, Sigma_y, u, Uy, by, Eh, Ehh):
    value = np.zeros((T,1))
    
    value += -1/2*(y.transpose(0,2,1) @ inv_covar_y @ y).reshape(T,1)
    
    Uu_plus_b = Uy @ u + by
    
    value += (y.transpose(0,2,1) @ inv_covar_y @ (Wy @ Eh 
                                                  + Uu_plus_b)).reshape(T,1)
    
    value += -1/2*np.trace(Wy.T @ inv_covar_y @ Wy @ Ehh, 
                           axis1=1, axis2 =2).reshape(T,1)

    value += -(Uu_plus_b.transpose(0,2,1) @ inv_covar_y 
               @ Wy @ Eh).reshape(T,1) 

    value += -1/2*( Uu_plus_b.transpose(0,2,1) @ inv_covar_y 
                    @ Uu_plus_b).reshape(T,1)

    log_term = T*(-yd/2*np.log(2*np.pi)-1/2*np.log(np.linalg.det(Sigma_y)) )
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
    
    print('gam_ent:',np.sum(value[:T,:,:]))
    print('omi_ent:',np.sum(value[T:2*T,:,:]))
    print('omf_ent:',np.sum(value[2*T:3*T,:,:]))
    print('omp_ent:',np.sum(value[3*T:4*T,:,:]))
    print('omo_ent:',np.sum(value[4*T:,:,:]))
    value = np.sum(value)
    print('entrpy_{}'.format(str), value)
    return value


def get_elbo(T, d, yd,  c_0, inv_covar_c, h_0, inv_covar_h, 
             Wi, Wf, Wp, Wo, y, inv_covar_y, Sigma_y,
             Ui, Uf, Up, Uo, Uy, u,
             bi, bf, bp, bo, by,
             Ec, Ecc_diags, Ecc_off_diags, Sigma_c, 
             Ev, Eh, Ehh_diags, Ehh,Sigma_h, 
             Ezi, Ezf, Ezp, Ezo, E_gamma, g_gamma, 
             Eomega_i, Eomega_f, Eomega_p, Eomega_o, 
             gi, gf, gp , go, L, h):

    elbo = elbo_y(T, yd, y, inv_covar_y, Sigma_y, u, Uy, by, Eh, Ehh)
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

def update_W_bar_y(T, d, ud, yd, y, Ex, Exx):
    
    sumExx = np.sum(Exx, axis = 0)
    
    #Construct yExT
    yExT  = (y[...,None]*Ex[:,None,:]).reshape(T,yd,d+ud+1)
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

#########################################################################
#####
seed = np.random.randint(0,10000)
#seed = 63#4687#  8604
np.random.seed(seed)
print('random_seed:',seed)

#elbo integration parameters
L = 10 #end of intergation interval
h = .001 #grid spacing


tol = .01 #paremeter difference tolerance
div = 400 #how often to compute elbo
T=10
d = 3 
ud = 2
yd = 1

var_c = .2
mu_c0 = .1
inv_covar_c = 1/var_c*np.ones((d,1))
c_0 = mu_c0*np.ones((d,1))

var_h = .3
mu_h0 = .3
inv_covar_h = 1/var_h*np.ones((d,1))
h_0 = mu_h0*np.ones((d,1))

u = np.random.uniform(-1,1, size=(T, ud, 1))

var_y = .4
Sigma_y = var_y*np.identity(yd)
inv_covar_y = 1/var_y*np.identity(yd)
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
E_gamma = np.random.uniform(0,1, size=(T,d,1))
Ev = np.random.uniform(0,1, size=(T,d,1))
Ezi = np.random.uniform(0,1, size=(T,d,1))
Ezf = np.random.uniform(0,1, size=(T,d,1))
Ezp = np.random.uniform(0,1, size=(T,d,1))
Ezo = np.random.uniform(0,1, size=(T,d,1))
Eomega_i = np.random.uniform(0,1, size=(T,d,1))
Eomega_f = np.random.uniform(0,1, size=(T,d,1))
Eomega_p = np.random.uniform(0,1, size=(T,d,1))
Eomega_o = np.random.uniform(0,1, size=(T,d,1))

'''
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
'''

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
W_bar_y_old = np.ones((yd, d+ud+1))*np.inf
Sigma_y_old = np.ones((yd, yd))*np.inf
W_bar_i_old = np.ones((d, d+ud+1))*np.inf
W_bar_f_old = np.ones((d, d+ud+1))*np.inf
W_bar_p_old = np.ones((d, d+ud+1))*np.inf
W_bar_o_old = np.ones((d, d+ud+1))*np.inf


diff = np.inf
diff_vec = []
elbo_vec = []
k_vec = []
k = 0


while diff > tol:
#for k in range(0,1):
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
    
    
    ###Elbo Calculation######
    if k % div ==0:
    
        elbo = get_elbo(T, d, yd, c_0, inv_covar_c, h_0, inv_covar_h, 
                        Wi, Wf, Wp, Wo, y, inv_covar_y, Sigma_y,
                        Ui, Uf, Up, Uo, Uy, u,
                        bi, bf, bp, bo, by,
                        Ec, Ecc_diags, Ecc_off_diags, Sigma_c, 
                        Ev, Eh, Ehh_diags, Ehh, Sigma_h,
                        Ezi, Ezf, Ezp, Ezo, E_gamma, g_gamma, 
                        Eomega_i, Eomega_f, Eomega_p, Eomega_o, 
                        gi, gf, gp , go, L, h)
        elbo_vec.append(elbo)
        k_vec.append(k)
        print('elbo:',elbo)
    
    ########################
    
    #Update Wy
        
    Ex_y, Exx_y = get_Ex_Exx(T, d, ud, Eh, Ehh, u)
    W_bar_y = update_W_bar_y(T, d, ud, yd, y, Ex_y, Exx_y)
    diff_list, W_bar_y_old = get_diff( W_bar_y, W_bar_y_old, diff_list)
    Wy, Uy, by = extract_W_weights(W_bar_y,  d, ud)
    by = by.reshape(yd,1)

    
    #Update Sigma_y
    Sigma_y = update_Sigma_y(T, yd, y, W_bar_y, Ex_y, Exx_y)
    diff_list, Sigma_y_old = get_diff( Sigma_y, Sigma_y_old, diff_list)
    inv_covar_y = np.linalg.inv(Sigma_y)
    
    
   
    ###Update W_stars###
    Eh_min, Ehh_min = get_Ehmin_Ehhmin(T, d, Eh, Ehh, h_0)
    Ex, Exx = get_Ex_Exx(T, d, ud, Eh_min, Ehh_min, u)
    
    
    
    #Update Wi
    
    W_bar_i = update_W_bar_star(T, d, ud, Ex, Exx, Eomega_i, Ezi)
    diff_list, W_bar_i_old = get_diff( W_bar_i, W_bar_i_old, diff_list)
    Wi, Ui, bi = extract_W_weights(W_bar_i,  d, ud)
    bi = bi.reshape(d,1)
    
    
    
    #Update Wf
    W_bar_f = update_W_bar_star(T, d, ud, Ex, Exx, Eomega_f, Ezf)
    diff_list, W_bar_f_old = get_diff( W_bar_f, W_bar_f_old, diff_list)
    Wf, Uf, bf = extract_W_weights(W_bar_f,  d, ud)
    bf = bf.reshape(d,1)
    
    
    #Update Wp
    W_bar_p = update_W_bar_star(T, d, ud, Ex, Exx, Eomega_p, Ezp)
    diff_list, W_bar_p_old = get_diff( W_bar_p, W_bar_p_old, diff_list)
    Wp, Up, bp = extract_W_weights(W_bar_p,  d, ud)
    bp = bp.reshape(d,1)
    
    
    #Update Wo
    W_bar_o = update_W_bar_star(T, d, ud, Ex, Exx, Eomega_o, Ezo)
    diff_list, W_bar_o_old = get_diff( W_bar_o, W_bar_o_old, diff_list)
    Wo, Uo, bo = extract_W_weights(W_bar_o,  d, ud)
    bo = bo.reshape(d,1)
    
    #convergence check
    diff = np.amax( diff_list )
    diff_vec.append(diff)
    print('diff:', diff)
    print('argmax_diff:',np.argmax(diff_list))
    

    k+=1
    print('iteration:',k)
    print(' ')



elbo = get_elbo(T, d, yd, c_0, inv_covar_c, h_0, inv_covar_h, 
                        Wi, Wf, Wp, Wo, y, inv_covar_y, Sigma_y,
                        Ui, Uf, Up, Uo, Uy, u,
                        bi, bf, bp, bo, by,
                        Ec, Ecc_diags, Ecc_off_diags, Sigma_c, 
                        Ev, Eh, Ehh_diags, Ehh, Sigma_h,
                        Ezi, Ezf, Ezp, Ezo, E_gamma, g_gamma, 
                        Eomega_i, Eomega_f, Eomega_p, Eomega_o, 
                        gi, gf, gp , go, L, h)
elbo_vec.append(elbo)
k_vec.append(k)
print('elbo:',elbo)



print('Ec:')
print(np.round(Ec,4))

'''
print('Ec prior:')
Ec_mod = np.zeros(T)
Ec_mod[0] = mu_0
Ec_mod[1:] = Ec[:-1]
print(Ezf*Ec_mod+Ezi*(2*Ezp-1))
print(' ')
'''
print('Ecc:')
print(np.round(Ecc,4))
print('Sigma_c')
print(np.round(Sigma_c,4))
print(' ')
print('Eh:',np.round(Eh,4))

''' 
print('Eh prior:')
print(Ezo*(2*Ev-1))
'''

print(' ')
print('Ehh')
print(np.round(Ehh,4))

print('Sigma_h:')
print(np.round(Sigma_h,4))
print(' ')
print('Ev:')
print(np.round(Ev,4))
'''
print('Ev prior:')
print(expit(2*Ec))
'''
print(' ')
print('E_gamma:')
print(np.round(E_gamma,4))
print('gg:')
print(np.round(g_gamma,4))  
print(' ')
print('Ezi')
print(np.round(Ezi,4))

'''
Eh_mod = np.zeros(T)
Eh_mod[0] = h_0
Eh_mod[1:] = Eh[:-1]
print('Ezi prior:')
print(expit(Wi*Eh_mod+Ui*u+bi))
'''

print(' ')
print('Ezf')
print(np.round(Ezf,4))
'''
print('Ezf prior:')
print(expit(Wf*Eh_mod+Uf*u+bf))
'''
print(' ')

print('Ezp')
print(np.round(Ezp,4))
'''
print('Ezp prior:')
print(expit(Wp*Eh_mod+Up*u+bp))
'''
print(' ')
print('Ezo')
print(np.round(Ezo,4))
'''
print('Ezo prior:')
print(expit(Wo*Eh_mod+Uo*u+bo))
'''
print(' ')
print('E_omega_i:')
print(np.round(Eomega_i,4))
print('E_omega_f:')
print(np.round(Eomega_f,4))
print('E_omega_p:')
print(np.round(Eomega_p,4))
print('E_omega_o:')
print(np.round(Eomega_o,4))
print('gi')
print( np.round(gi,d))
print('gf')
print( np.round(gf,4))
print('gp')
print( np.round(gp,4))
print('go')
print( np.round(go,4))

print('Sigma_y')
print(Sigma_y)

'''
print('Wbary')
print(np.round(W_bar_y,3))

print('Wbar_i')
print(np.round(W_bar_i,3))

print('Wbar_f')
print(np.round(W_bar_f,3))

print('Wbar_p')
print(np.round(W_bar_p,3))

print('Wbar_o')
print(np.round(W_bar_o,3))
'''
print('elbo:', elbo)


print('random_seed:',seed)




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




