import numpy as np
from scipy.special import expit
import sys
import time
import matplotlib.pyplot as plt
from PG_int import qdf_log_pdf, entropy_q, qdf_log_pdf_vec, entropy_q_vec
from scipy import integrate
import os

def generate_sample(T,d, yd, u, c0, Sigma_c, h0, Sigma_h, Sigma_y,
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

    

    zi[0,:] = np.random.binomial(1, expit(Wi @ h0 + Ui @ u[0,:] + bi))
    zf[0,:] = np.random.binomial(1, expit(Wf @ h0 + Uf @ u[0,:] + bf))
    zp[0,:] = np.random.binomial(1, expit(2*(Wp @ h0 + Up @ u[0,:] + bp)))
    zo[0,:] = np.random.binomial(1, expit(Wo @ h0 + Uo @ u[0,:] + bo))


    cg[0,:] = np.random.normal(zf[0,:]*c0+zi[0,:]*(2*zp[0,:]-1),
                            np.sqrt(Sigma_c))

    
    v[0,:] = np.random.binomial(1,  expit(2*cg[0,:]))
    hg[0,:] = np.random.normal(zo[0,:]*(2*v[0,:]-1), np.sqrt(Sigma_h)) 
    yg[0,:] = np.random.multivariate_normal(Wy @ hg[0,:] + Uy @ u[0,:] + by,
                                           Sigma_y)
    for t in range(1,T):
        zi[t,:] = np.random.binomial(1, expit(Wi @ hg[t-1,:]+Ui @ u[t,:]+bi))
        zf[t,:] = np.random.binomial(1, expit(Wf @ hg[t-1,:]+Uf @ u[t,:]+bf))
        zp[t,:] = np.random.binomial(1, expit(2*(Wp @ hg[t-1,:]+
                                                 Up @ u[t,:]+bp)))
        zo[t,:] = np.random.binomial(1, expit(Wo @ hg[t-1,:]+Uo @ u[t,:]+bo))
        cg[t,:] = np.random.normal(zf[t,:]*cg[t-1,:]+zi[t,:]*(2*zp[t,:]-1),
                                   np.sqrt(Sigma_c))
        v[t,:] = np.random.binomial(1,  expit(2*cg[t,:]))
        hg[t,:] = np.random.normal(zo[t,:]*(2*v[t,:]-1), 
                                   np.sqrt(Sigma_h)) 
        yg[t,:] = np.random.multivariate_normal(Wy @ hg[t,:] + 
                                               Uy @ u[t,:] + by,
                                               Sigma_y)
    return yg, cg, hg, v, zi, zf, zp, zo


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

def Lambda_h_mP_op( Ez_star, Eomega_star, W_star, 
                  U_star, b_star, u):
    value = 2*W_star.T @ (Ez_star[1:,:,:]-1/2)
    value  += -4*W_star.T @ (Eomega_star[1:,:,:]
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
        Lambda[t,:,:] += 4*Lambda_h_op(t, d, Eomega_p, Wp)
        Lambda[t,:,:] += Lambda_h_op(t, d, Eomega_o, Wo)

    #Lambda_m shape = (T,d,1)
    Lambda_m = np.zeros((T,d,1))
    Lambda_m += (inv_covar_y @ Wy).T @ (y-(Uy @ u + by))
    
    Lambda_m += inv_covar * ( Ezo*(2*Ev-1) )
    Lambda_m[:-1,:,:] += Lambda_h_m_op(Ezi, Eomega_i, Wi, Ui, bi, u)
    Lambda_m[:-1,:,:] += Lambda_h_m_op(Ezf, Eomega_f, Wf, Uf, bf, u)
    Lambda_m[:-1,:,:] += Lambda_h_mP_op(Ezp, Eomega_p, Wp, Up, bp, u)
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
    g = np.sqrt(4*value)
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
    
    value = W_Eh_z_update(2*Wp, 2*Up, 2*bp, u, Eh, h_0, value)
    
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
#seed = 6189 #9765#4687#  8604
np.random.seed(seed)
print('random_seed:',seed)

#elbo integration parameters
comp_elbo = False
L = 10    #end of integration interval
h = .01   #grid spacing
div = 60 #how often to compute elbo


##### Data ##########################################################
#u = np.random.uniform(-1,1, size=(T, ud, 1))
#y = np.random.uniform(-1,1, size = (T, yd,1))

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
var_c = .3
var_h = .3
N = 1     #Number of Monte Carlo samples
tol = .01 #convergence check



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
wfile = 'weights/LSTM_d10_eps10_lr0.001_end200_1567714051.980048.npy'
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
'''
##Need to change code!
#Uy = np.random.uniform(-1,1, size=(yd,ud))




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


timestamp = time.time()
path = 'images/{}'.format(timestamp)


os.mkdir(path)


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

    gp, Eomega_p = update_q_omega_P(T,d,Eh, Ehh, h_0, 
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
    if comp_elbo == True:
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
    W_bar_p = update_W_bar_star(T, d, ud, Ex, Exx, 2*Eomega_p, Ezp)
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
    

    if k % 50 == 0:
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
                                         c0.reshape(d), Sig_c.reshape(d), 
                                         h0.reshape(d), Sig_h.reshape(d), 
                                         Sigma_y,
                                         Wy, Wi, Wf, Wp, Wo, 
                                         Uy, Ui, Uf, Up, Uo, 
                                         by.reshape(yd), bi.reshape(d), 
                                         bf.reshape(d), bp.reshape(d), 
                                         bo.reshape(d))
            y_tr_gen_vec[j,:] += y_tr_gen.reshape(yd)
            c_arr[j,:] = c0.reshape(d)
            h_arr[j,:] = h0.reshape(d)


        r = np.arange(stop, stop+T_new,dt)
        plt.plot(r, y_tr_gen_vec, label = '{}'.format(diff))
        #plt.savefig(path+ '/d{}__varh{}_varc{}.png'.format(  d, var_h,  var_c))
        



    k+=1
    print('iteration:',k)
    print(' ')


plt.legend()
plt.savefig(path+ '/d{}__varh{}_varc{}.png'.format(  d, var_h,  var_c))
plt.close()

'''
np.save('weights/{}_Wbary'.format(timestamp), W_bar_y)
np.save('weights/{}_Wbari'.format(timestamp), W_bar_i)
np.save('weights/{}_Wbarf'.format(timestamp), W_bar_f)
np.save('weights/{}_Wbarp'.format(timestamp), W_bar_p)
np.save('weights/{}_Wbaro'.format(timestamp), W_bar_o)
np.save('weights/{}_Sigmay'.format(timestamp), Sigma_y)
np.save('weights/{}_Eh'.format(timestamp), Eh)
np.save('weights/{}_Sigmah'.format(timestamp), Sigma_h)
np.save('weights/{}_Ec'.format(timestamp), Ec)
np.save('weights/{}_Sigmac'.format(timestamp), Sigma_c)
'''


#Final elbo computation
if comp_elbo == True:
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
    
    print('random_seed:',seed)
    
    plt.plot(k_vec, elbo_vec)
    plt.xlabel('Iteration')
    plt.ylabel('ELBO')
    plt.title('ELBO convergence')
    plt.savefig('images/ELBO__T{}__dt{}_d{}_N{}_varh{}_varc{}.png'.format(end,
                                                   dt, d, N, var_h,var_c))
    plt.show()
    plt.close()

    plt.plot(np.arange(k), diff_vec)
    plt.xlabel('Iteration')
    plt.ylabel('Max Parameter Difference')
    plt.savefig('Error.png')
    plt.show()
    plt.close()



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

'''
print('Ecc:')
print(np.round(Ecc,4))
'''

print('Sigma_c')
print(np.round(Sigma_c,4))
print(' ')
print('Eh:',np.round(Eh,4))

''' 
print('Eh prior:')
print(Ezo*(2*Ev-1))
'''
'''
print(' ')
print('Ehh')
print(np.round(Ehh,4))
'''
print('Sigma_h:')
print(np.round(Sigma_h,4))
print(' ')
print('Ev:')
print(np.round(Ev,4))
'''
print('Ev prior:')
print(expit(2*Ec))
'''
'''
print(' ')
print('E_gamma:')
print(np.round(E_gamma,4))
print('gg:')
print(np.round(g_gamma,4))  
'''

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
'''
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
                                         c_0.reshape(d), Sig_c.reshape(d), 
                                         h_0.reshape(d), Sig_h.reshape(d), 
                                         Sigma_y,
                                         Wy, Wi, Wf, Wp, Wo, 
                                         Uy, Ui, Uf, Up, Uo, 
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
                                         Uy, Ui, Uf, Up, Uo, 
                                         by.reshape(yd), bi.reshape(d), 
                                         bf.reshape(d), bp.reshape(d), 
                                         bo.reshape(d))
y_test_vec += y_gen
'''

'''
##Test code below using learned model on new data with previous y as 
#input, but one at a time
y_test_vec2 = np.zeros((T_new,yd))
h_arr = np.zeros((T_new,d))
c_arr= np.zeros((T_new,d))

for j in range(0,T_new):
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
                                         Uy, Ui, Uf, Up, Uo, 
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
        y_tr_gen = y[-1,:,:].reshape(1,yd)
        
        c_0 = Ec[-1,:,0].reshape(d) 
        h_0 = Eh[-1,:,0].reshape(d)
        
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


