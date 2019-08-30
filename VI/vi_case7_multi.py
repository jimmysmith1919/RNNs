import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
from E_PG import qdf_log_pdf, entropy_q
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

    #Construct Precision Diagonal
    diag = np.zeros((d,T,1))
    diag += inv_covar+4*E_gamma    
    diag[:,:-1,:] += inv_covar*Ezf_re[:,1:,:]
    Lambda +=  diag*np.identity(T)

    '''
    diag1 = inv_covar*np.identity(T)
    diag2 = np.zeros(T)
    diag2[:-1] = diag1[1:]*Ezf[1:]
    Lambda = np.diag(diag1+diag2+4*E_gamma)
    '''
    #Construct Precision Off Diagonals
    off_diag = -inv_covar*Ezf_re[:,1:,:]
    Lambda[:,:-1,1:] += off_diag*np.identity(T-1)
    Lambda[:,1:,:-1] += off_diag*np.identity(T-1)
    '''
    #Construct Precision Off diagonals
    off_diag = -1/(np.diag(covar)[1:])*Ezf[1:]
    Lambda += np.diag(off_diag, k=1)
    Lambda += np.diag(off_diag, k=-1)
    '''

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
    
    '''
    Lambda_m[0] = 1/covar[0,0]*mu_0*Ezf[0]
    Lambda_m += diag1*Ezi*(2*Ezp-1)
    Lambda_m += 2*(Ev-1/2)
    Lambda_m[:-1] += -diag1[1:]*Ezi[1:]*Ezf[1:]*(2*Ezp[1:]-1)
    '''
    return  Lambda, Lambda_m

def get_moments(Lambda, Lambda_m):
    Sigma = np.linalg.inv(Lambda)
    Em = Sigma @ Lambda_m
    Emm = (Em[...,None]*Em[:,None,:]).reshape(Sigma.shape)+Sigma
    return Em, Emm

def get_diags(Exx,d,T):
    '''Extracts diagonals and 1 off diagonals from dxTxT array '''
    diags = np.zeros((d,T))
    off_diags = np.zeros((d,T-1))
    for i in range(0,d):
        diags[i,:] = np.diag(Ecc[i,:,:])
        off_diags[i,:] = np.diag(Ecc[i,:,:], k=-1)
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


def update_qh(T,d,h_0, inv_covar, Wi, Wf, Wp, Wo, 
              u, Ui, Uf, Up, Uo, Ev, Eomega_i, Eomega_f, 
              Eomega_p, Eomega_o, Ezi, Ezf, Ezp, Ezo):
    
    #Construct Precision_d matrix
    #Lambda shape (T,d,d)
    Lambda = np.zeros((T,d,d))
    Lambda += inv_covar*np.identity(d)
    for t in range(0,T-1):
        Lambda[t,:,:] += Lambda_h_op(t, Eomega_i, Wi)
        Lambda[t,:,:] += Lambda_h_op(t, Eomega_f, Wf)
        Lambda[t,:,:] += Lambda_h_op(t, Eomega_p, Wp)
        Lambda[t,:,:] += Lambda_h_op(t, Eomega_o, Wo)
    

    #Lambda_m shape = (T,d,1)
    Lambda_m = np.zeros((T,d,1))
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
        value[0,:,:] += tr_val
        tr_val = np.trace(wwT @ Ehh[:-1,:,:], 
                          axis1=1, axis2=2).reshape(d,1)
        value[1:,i,:] += tr_val
    
    Uu_plus_b = U_star @ u + b_star
    
    value[0,:,:] += (2*W_star @ h_0)*Uu_plus_b[0,:,:]
    value[1:,:,:] += (2*W_star @ Eh[:-1,:,:])*Uu_plus_b[1:,:,:]

    value += Uu_plus_b**2
    
    g = np.sqrt(value)
    E_omega = 1/(2*g)*np.tanh(g/2)
    '''
    g = np.zeros(len(Ehh))
    g[0] = np.sqrt( (W_star**2)*h_0**2 +
                    2*(U_star*u[0]+b_star)*W_star*h_0 +
                    (U_star*u[0]+b_star)**2 )
    g[1:] = np.sqrt( W_star**2*Ehh[:-1] +
                     2*(U_star*u[1:]+b_star)*W_star*Eh[:-1] +
                     (U_star*u[1:]+b_star)**2 )
    b = np.ones(len(g))
    E_omega = b/(2*g)*np.tanh(g/2)'''
    return g, E_omega
##############################################################

def update_qv(Eh, inv_covar, Ec, Ezo):
    value = 2*inv_covar*Eh*Ezo+2*Ec
    '''diag = np.diag(covar)
    value = 2/diag*Eh*Ezo+2*Ec'''
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

    '''
     Lambda_m[:,0,:] += (inv_covar*c_0*(
            Ezf_re[:,0,:].reshape(d,1,1) )).reshape(d,1)
    '''

    '''
    diag = np.diag(covar)
    ones = np.ones(len(Ec))
    
    value = 1/diag*Ec*(2*Ezp-ones)
    
    Ec_minus = np.zeros(len(Ec))
    Ec_minus[0] = mu_0
    Ec_minus[1:] = Ec[:-1]
    value += -1/diag*Ezf*Ec_minus*(2*Ezp-1)-(1/2)*(1/diag)
    
    value = W_Eh_z_update(Wi, Ui, bi, u, Eh, h_0, value)
    '''
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
    
    '''
    diag = np.diag(covar)
    
    Ecc_minus_1 = np.zeros(len(Ec))
    Ecc_minus_1[0] = Ec[0]*mu_0
    Ecc_minus_1[1:] = np.diag(Ecc, k=-1)
    
    value = 1/diag*Ecc_minus_1

    Ec_min1_sq = np.zeros(len(Ec))
    Ec_min1_sq[0] = mu_0**2
    Ec_min1_sq[1:] = np.diag(Ecc)[:-1]

    value += -1/2*1/diag*Ec_min1_sq

    Ec_min1 = np.zeros(len(Ec))
    Ec_min1[0] = mu_0
    Ec_min1[1:] = Ec[:-1]

    value += -1/diag*Ec_min1*Ezi*(2*Ezp-1)
    
    value = W_Eh_z_update(Wf, Uf, bf, u, Eh, h_0, value)
    '''
    return expit(value)

def update_zp(c_0, inv_covar, Wp, Up, bp, u, h_0, Eh, Ec, Ezi, Ezf):
    value = 2*inv_covar*Ec*Ezi

    value[0,:,:] += -2*inv_covar*Ezf[0,:,:]*c_0*Ezi[0,:,:]
    value[1:,:,:] += -2*inv_covar*Ezf[1:,:,:]*Ec[:-1,:,:]*Ezi[1:,:,:]
    
    value = W_Eh_z_update(Wp, Up, bp, u, Eh, h_0, value)
    
    '''diag = np.diag(covara)
    
    value = 2/diag*Ec*Ezi

    Ec_minus = np.zeros(len(Ec))
    Ec_minus[0] = mu_0
    Ec_minus[1:] = Ec[:-1]

    value += -2/diag*Ezf*Ec_minus*Ezi

    value = W_Eh_z_update(Wp, Up, bp, u, Eh, h_0, value)'''
    return expit(value)

def update_zo(h_0, inv_covar, Wo, Uo, bo, u, Eh, Ev):
    value = inv_covar*Eh*(2*Ev-1) - 1/2*inv_covar
    value = W_Eh_z_update(Wo, Uo, bo, u, Eh, h_0, value)
    '''
    diag = np.diag(covar)
    
    value = 1/diag*Eh*(2*Ev-1)-1/2*1/diag

    value = W_Eh_z_update(Wo, Uo, bo, u, Eh, h_0, value)
    '''
    return expit(value)

                    

###########################################################
##ELBO calculation###


def elbo_c(T, d, c_0, inv_covar, Ec, Ecc_diags, Ecc_off_diags):
    value = np.zeros((T,d,1))
    value += np.log(np.sqrt(1/(2*np.pi)*inv_covar))

    value += -1/2*inv_covar*Ecc_diags

    value[0,:,:] += inv_covar*Ezf[0,:,:]*Ec[0,:,:]*c_0 
    value[1:,:,:] += inv_covar*Ezf[1:,:,:]*Ec[:-1,:,:]*Ecc_off_diags

    value += inv_covar*Ezi*(2*Ezp-1)*Ec

    value[0,:,:] += -1/2*inv_covar*Ezf[0,:,:]*c_0**2
    value[1:,:,:] += -1/2*inv_covar*Ezf[1:,:,:]*Ecc_diags[:-1,:,:]

    value[0,:,:] += -inv_covar*Ezi[0,:,:]*Ezf[0,:,:]*(2*Ezp[0,:,:]
                                                      -1)*c_0
    value[1:,:,:] += -inv_covar*Ezi[1:,:,:]*Ezf[1:,:,:]*(2*Ezp[1:,:,:]
                                                      -1)*Ec[:-1,:,:]
    value += -1/2*inv_covar*Ezi

    '''
    diag = np.diag(covar)
    Ecc_diag = np.diag(Ecc)

    value = np.log( 1/np.sqrt(2*np.pi*diag) )
    value += -1/2*( 1/diag )*Ecc_diag

    term = 1/diag
    term[0] *= m[0]*mu_0*Ezf[0]
    term[1:] *= np.diag(Ecc,-1)*Ezf[1:]
    value += term

    value += 1/diag*Ezi*(2*Ezp-1)*m
    
    term = -1/2*1/diag
    term[0] *= mu_0**2*Ezf[0]
    term[1:] *= np.diag(Ecc)[:-1]*Ezf[1:]
    value += term

    term = -1/diag*Ezi*Ezf*(2*Ezp-1)
    term[0] *= mu_0
    term[1:] *= m[:-1]
    value += term
    
    value += -1/2*1/diag*Ezi
    '''
    
    print('elbo_c:', np.sum(value))
    return np.sum(value)

def elbo_h(h_0, covar, Eh, Ehh, Ezo, Ev):
    diag = np.diag(covar)

    value = np.log( 1/np.sqrt(2*np.pi*diag) )
    value += -1/2*( 1/diag )*Ehh
    value += 1/diag*Ezo*(2*Ev-1)*Eh
    value += -1/2*1/diag*Ezo
    print('elbo_h:', np.sum(value))
    return np.sum(value)

    
def elbo_z_star(Ez,Eh, h_0, W_star, U_star, b_star, u, star):
    
    mod_Eh = np.zeros(len(Eh))
    mod_Eh[0] = h_0
    mod_Eh[1:] = Eh[:-1]

    value = np.zeros(len(Ez))
    value += np.log(1/2) 
    value += (Ez-1/2)*(W_star*mod_Eh+U_star*u+b_star)
    print('elbo_{}'.format(star), np.sum(value))
    return np.sum(value)


def elbo_v(Ev, Ec):
    ones = np.ones(len(Ev))
    value = np.log(1/2)*ones + 2*(Ev-1/2)*Ec
    print('elbo_v:', np.sum(value))
    return np.sum(value)

def elbo_gamma1(E_gamma, Ecc):
    value = -1/2*E_gamma*4*np.diag(Ecc)
    print('elbo_gamma1:', np.sum( value) )
    return np.sum( value )

def elbo_gamma2(Ecc):
    Ecc_diag = np.diag(Ecc)
    value = 0
    for t in range(0,len(Ecc_diag)):
        value += integrate.quad(qdf_log_pdf, 0, np.inf,
                           args=(1,0, np.sqrt(4*Ecc_diag[t])),
                           epsabs=1e-1, epsrel = 0)[0]
    print('elbo_gamma', value)
    return value

def elbo_omega_star_1(Eomega_star, g_star, star):
    value = -1/2*Eomega_star*g_star**2
    print('elbo_omega1_{}:'.format(star), np.sum( value) )
    return np.sum( value )

def elbo_omega2_star(g_star, star):
    value = 0
    for t in range(0,len(g_star)):
        value += integrate.quad(qdf_log_pdf, 0, np.inf,
                           args=(1,0, g_star[t]),
                           epsabs=1e-1, epsrel = 0)[0]
    print('elbo_omega2_{}'.format(star), value)
    return value


def entropy_c(T, Ec, Ecc):
    #need to make more efficient
    Sigma = Ecc-np.outer(Ec,Ec)
    print('Entropy_c:', T/2*(1+np.log(2*np.pi))+
          1/2*np.log( np.linalg.det(Sigma) ))
    return T/2*(1+np.log(2*np.pi))+1/2*np.log( np.linalg.det(Sigma) )


def entropy_h(T, Eh, Ehh):
    ###not using last h for this case
    Eh = Eh[:-1]
    Ehh = Ehh[:-1]
    ###

    Sigma = Ehh-Eh**2
    print('Entropy_h:', T/2*(1+np.log(2*np.pi))+
          1/2*np.log( np.prod(Sigma) ))
    return T/2*(1+np.log(2*np.pi))+1/2*np.log( np.prod(Sigma) )

def entropy_Bern(p, str):
    ones = np.ones(len(p))
    
    check1 = np.argwhere(np.isclose(p, np.zeros(len(p))))
    check2 = np.argwhere(np.isclose(p, ones))
    value = -p*np.log(p)-(ones-p)*np.log(ones-p)
    value[check1] = 0
    value[check2] = 0
    print('entropy_{}:'.format(str), np.sum(value))
    return np.sum(value)


def entropy_gamma(Ecc):
    Ecc_diag = np.diag(Ecc)
    value = 0
    for t in range(0,len(Ecc_diag)):
        value += integrate.quad(entropy_q, 0, np.inf, 
                                args=(1,np.sqrt(4*Ecc_diag[t])),
                                epsabs=1e-1, epsrel=0)[0]
    print('gam_entrpy:', value)
    return value

def entropy_omega_star(g_star, star):
    value = 0
    for t in range(0,len(g_star)):
        value += integrate.quad(entropy_q, 0, np.inf, 
                                args=(1, g_star[t]),
                                epsabs=1e-1, epsrel=0)[0]
    print('omega_{}_entrpy:'.format(star), value)
    return value

def get_elbo(T,  mu_0, covar_c, h_0, covar_h, Wi, Wf, Wp, Wo, 
             Ui, Uf, Up, Uo, u, 
             Ec, Ecc, Ev, Eh, Ehh, Ezi, Ezf, Ezp, Ezo, E_gamma, 
             Eomega_i, Eomega_f, Eomega_p, Eomega_o, 
             gi, gf, gp , go):
    #NOTE: currently excludes several PG terms
    elbo = elbo_c(mu_0, Ec, covar_c, Ecc)
    elbo += elbo_v(Ev, Ec)
    elbo += elbo_h(h_0, covar_h, Eh, Ehh, Ezo, Ev)


    elbo += elbo_z_star(Ezi,Eh, h_0, Wi, Ui, bi, u, 'zi')
    elbo += elbo_z_star(Ezf,Eh, h_0, Wf, Uf, bf, u, 'zf')
    elbo += elbo_z_star(Ezp,Eh, h_0, Wp, Up, bp, u, 'zp')
    elbo += elbo_z_star(Ezo,Eh, h_0, Wo, Uo, bo, u, 'zo')
    
    elbo += elbo_gamma1(E_gamma, Ecc)
    elbo += elbo_gamma2(Ecc)

    elbo += elbo_omega_star_1(Eomega_i, gi, 'i')
    elbo += elbo_omega_star_1(Eomega_f, gf, 'f')
    elbo += elbo_omega_star_1(Eomega_p, gp, 'p')
    elbo += elbo_omega_star_1(Eomega_o, go, 'o')

    
    ###
    #multiply times 4 since Wi=Wf=Wp input to save time
    elbo_omega2 = elbo_omega2_star(gi, 'i')
    elbo += elbo_omega2*4
    
    '''
    elbo += elbo_omega2_star(gi, 'i')
    elbo += elbo_omega2_star(gf, 'f')
    elbo += elbo_omega2_star(gp, 'p')
    elbo += elbo_omega2_star(go, 'o')
    '''
    ###

    elbo += entropy_c(T, Ec, Ecc)
    elbo += entropy_Bern(Ev, 'v')
    elbo += entropy_h(T, Eh, Ehh)

    elbo += entropy_Bern(Ezi, 'zi')
    elbo += entropy_Bern(Ezf, 'zf')
    elbo += entropy_Bern(Ezp, 'zp')
    elbo += entropy_Bern(Ezo, 'zo')

    elbo += entropy_gamma(Ecc)

    
    ### multiply times 4 since same input to save time
    ent_omega = entropy_omega_star(gi, 'i')
    elbo += ent_omega*4
    
    '''
    elbo += entropy_omega_star(gi, 'i')
    elbo += entropy_omega_star(gf, 'f')
    elbo += entropy_omega_star(gp, 'p')
    elbo += entropy_omega_star(go, 'o')
    '''
    ###

    return elbo

def get_diff(param, param_old, diff_list):
    param_diff = np.amax(np.absolute(param-param_old))
    diff_list.append(param_diff)
    param_old = param
    return diff_list, param_old

#####
np.random.seed(0)
T=4
d = 3
ud = 2

mu_c0 = .1
mu_h0 = .2

#inv_covar_c = 1/.2*np.ones((d,1,1))
inv_covar_c = 1/.2*np.ones((d,1))
c_0 = mu_c0*np.ones((d,1))

inv_covar_h = 1/.3*np.ones((d,1))#identity(d)


print('inv_covar_h')
print(inv_covar_h)

h_0 = mu_h0*np.ones((d,1))

u = .4*np.ones((T, ud, 1))

Wi = np.random.uniform(-1,1, size=(d,d))
Wf = np.random.uniform(-1,1, size=(d,d))
Wp = np.random.uniform(-1,1, size=(d,d))
Wo = np.random.uniform(-1,1, size=(d,d))

Ui = np.random.uniform(-1,1, size=(d,ud))
Uf = np.random.uniform(-1,1, size=(d,ud))
Up = np.random.uniform(-1,1, size=(d,ud))
Uo = np.random.uniform(-1,1, size=(d,ud))

bi = np.random.uniform(-1,1, size=(d,1))
bf = np.random.uniform(-1,1, size=(d,1))
bp = np.random.uniform(-1,1, size=(d,1))
bo = np.random.uniform(-1,1, size=(d,1))

'''
W_i = np.concatenate((Wi, Ui, bi), axis=1)
W_f = np.concatenate((Wf, Uf, bf), axis=1)
W_p = np.concatenate((Wp, Up, bp), axis=1)
W_o = np.concatenate((Wo, Uo, bo), axis=1)
'''



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
E_gamma = .3*np.ones((d,T,1))
Ev = .8*np.ones((T,d,1))
Ezi = .3*np.ones((T,d,1))
Ezf = .3*np.ones((T,d,1))
Ezf[2:,1:,:] = .4
Ezf[1,1,:] = .2
Ezf[0,0,:] = .6
Ezp = .3*np.ones((T,d,1))
Ezo = .3*np.ones((T,d,1))
Eomega_i = .3*np.ones((T,d,1))
Eomega_f = .3*np.ones((T,d,1))
Eomega_p = .3*np.ones((T,d,1))
Eomega_o = .3*np.ones((T,d,1))


Lambda_c, Lambda_c_m  = update_qc(T,d, c_0, inv_covar_c, 
                               Ezi, Ezf, Ezp, Ev, E_gamma)

Ec, Ecc = get_moments(Lambda_c, Lambda_c_m)
Ec = to_dxT(T,d, Ec)
print('Ecc')
print(Ecc)
print()
Ecc_diags, Ecc_off_diags = get_diags(Ecc,d,T)
Ecc_diags = np.reshape(Ecc_diags.ravel('F'),(T,d,1))
Ecc_off_diags = np.reshape(Ecc_off_diags.ravel('F'), (T-1,d,1))


Lambda_h, Lambda_h_m = update_qh(T,d,h_0, inv_covar_h, Wi, Wf, Wp, 
              Wo, u, Ui, Uf, Up, Uo, Ev, Eomega_i, Eomega_f, 
              Eomega_p, Eomega_o, Ezi, Ezf, Ezp, Ezo)

Eh, Ehh = get_moments(Lambda_h, Lambda_h_m)

Ezi = update_zi(T,d, c_0, inv_covar_c, Wi, Ui, bi, u, h_0, 
              Eh, Ec, Ezp, Ezf)


Ezf = update_zf(T,d, c_0, inv_covar_c, Wf, Uf, bf, u, 
              h_0, Eh,  Ec, Ecc_diags, Ecc_off_diags, Ezi, Ezp)
Ezp = update_zp(c_0, inv_covar_c, Wp, Up, bp, u, 
                h_0, Eh, Ec, Ezi, Ezf)
Ezo = update_zo(h_0, inv_covar_h, Wo, Uo, bo, u, Eh, Ev)

Ev = update_qv(Eh, inv_covar_h, Ec, Ezo)

g, E_gamma = update_q_gamma(Ecc_diags)


g, E_omega = update_q_omega_star(T,d,Eh, Ehh, h_0, Wi, Ui, bi, u)


elbo = elbo_c(T, d, c_0, inv_covar_c, Ec, Ecc_diags, Ecc_off_diags)

'''
Ec_old = np.ones((T,d))*np.inf
Ecc_old = np.ones((d,T,T))*np.inf
Eh_old = np.ones((d,T))*np.inf
Ehh_old = np.ones((d,T))*np.inf
Ev_old = np.ones((d,T))*np.inf
gg_old = np.ones((d,T))*np.inf
Ezi_old = np.ones((d,T))*np.inf
Ezf_old = np.ones((d,T))*np.inf
Ezp_old = np.ones((d,T))*np.inf
Ezo_old = np.ones((d,T))*np.inf
gi_old = np.ones((d,T))*np.inf
gf_old = np.ones((d,T))*np.inf
gp_old = np.ones((d,T))*np.inf
go_old = np.ones((d,T))*np.inf


diff = np.inf
tol = .001


diff_vec = []
elbo_vec = []

k = 0




while diff > tol:
    diff_list = []

    #update qc
    Lambda_c, Lambda_c_m = update_qc(T, mu_0, covar_c, 
                                     Ezi, Ezf, Ezp, Ev, E_gamma)
    Ec, Ecc = get_c_expects(Lambda_c, Lambda_c_m) 
    diff_list, Ec_old = get_diff( Ec, Ec_old, diff_list)

    diff_list, Ecc_old = get_diff( Ecc, Ecc_old, diff_list)
    
    #update q_h:
    Lambda_h, Lambda_h_m = update_qh(T, h_0, covar_h, Wi, Wf, Wp, Wo, 
              u, Ui, Uf, Up, Uo, Ev, Eomega_i, Eomega_f, 
              Eomega_p, Eomega_o, Ezi, Ezf, Ezp, Ezo)
    Eh, Ehh = get_h_expects(Lambda_h, Lambda_h_m)

    diff_list, Eh_old = get_diff( Eh, Eh_old, diff_list)
    diff_list, Ehh_old = get_diff( Ehh, Ehh_old, diff_list)

    #update qv
    Ev = update_qv(Eh, covar_h, Ec, Ezo)
    diff_list, Ev_old = get_diff( Ev, Ev_old, diff_list)
        
    #update q_gamma
    b, gg, E_gamma = update_q_gamma(Ecc)
    diff_list, gg_old = get_diff( gg, gg_old, diff_list)
    

    #update q_omegas
    gi, Eomega_i = update_q_omega_star(Eh, Ehh, h_0, 
                                           Wi, Ui, bi, u)
    diff_list, gi_old = get_diff( gi, gi_old, diff_list)


    gf, Eomega_f = update_q_omega_star(Eh, Ehh, h_0, 
                                           Wf, Uf, bf, u)
    diff_list, gf_old = get_diff( gf, gf_old, diff_list)

    gp, Eomega_p = update_q_omega_star(Eh, Ehh, h_0, 
                                           Wp, Up, bp, u)
    diff_list, gp_old = get_diff( gp, gp_old, diff_list)

    go, Eomega_o = update_q_omega_star(Eh, Ehh, h_0, 
                                           Wo, Uo, bo, u)
    diff_list, go_old = get_diff( go, go_old, diff_list)
    
    #update q_zi
    Ezi = update_zi(mu_0, covar_c, Wi, Ui, bi, u, 
                    h_0, Eh, Ec, Ezp, Ezf)
    diff_list, Ezi_old = get_diff( Ezi, Ezi_old, diff_list)

    #update q_zf
    Ezf = update_zf(mu_0, covar_c, Wf, Uf, bf, u,
                    h_0, Eh,  Ec, Ecc, Ezi, Ezp)
    diff_list, Ezf_old = get_diff( Ezf, Ezf_old, diff_list)

    #update q_zp
    Ezp = update_zp(mu_0, covar_c, Wp, Up, bp, u, 
                    h_0, Eh, Ec, Ezi, Ezf)
    diff_list, Ezp_old = get_diff( Ezp, Ezp_old, diff_list)
    
    #update q_zo
    Ezo = update_zo(h_0, covar_h, Wo, Uo, bo, u, Eh, Ev)
    diff_list, Ezo_old = get_diff( Ezo, Ezo_old, diff_list)


    #convergence check
    diff = np.amax( diff_list )
    diff_vec.append(diff)

    
    elbo = get_elbo(T,  mu_0, covar_c, h_0, covar_h, Wi, Wf, Wp, Wo, 
             Ui, Uf, Up, Uo, u, 
             Ec, Ecc, Ev, Eh, Ehh, Ezi, Ezf, Ezp, Ezo, E_gamma, 
             Eomega_i, Eomega_f, Eomega_p, Eomega_o, 
             gi, gf, gp, go)

    elbo_vec.append(elbo)
    print(' ')
    
    k+=1


elbo = get_elbo(T,  mu_0, covar_c, h_0, covar_h, Wi, Wf, Wp, Wo, 
             Ui, Uf, Up, Uo, u, 
             Ec, Ecc, Ev, Eh, Ehh, Ezi, Ezf, Ezp, Ezo, E_gamma, 
             Eomega_i, Eomega_f, Eomega_p, Eomega_o, 
             gi, gf, gp, go)

elbo_vec.append(elbo)
print(' ')



print('Ec:', Ec)
print('Ec prior:')
Ec_mod = np.zeros(T)
Ec_mod[0] = mu_0
Ec_mod[1:] = Ec[:-1]
print(Ezf*Ec_mod+Ezi*(2*Ezp-1))
print(' ')
#print('Lambda_c')
#print(Lambda_c)
#print('Lambda_c_m:', Lambda_c_m)
print('Ecc:')
print(Ecc)
Sigma = Ecc-np.outer(Ec,Ec)
#print('Sigma:')
#print(Sigma)
print(' ')
print('Eh:',Eh) 
print('Eh prior:')
print(Ezo*(2*Ev-1))
print(' ')
print('Ehh', Ehh)
#print('Lambda_h')
#print(Lambda_h)
#Sigma_h = Ehh-Eh**2
#print('Sigma_h:', Sigma_h)
#print('Lamb_h_m')
#print(Lambda_h_m)
print(' ')
print('Ev:', Ev)
print('Ev prior:')
print(expit(2*Ec))
print(' ')
print('E_gamma:', E_gamma)
print('gg:', gg)  
print(' ')
print('Ezi', Ezi)
Eh_mod = np.zeros(T)
Eh_mod[0] = h_0
Eh_mod[1:] = Eh[:-1]
print('Ezi prior:')
print(expit(Wi*Eh_mod+Ui*u+bi))
print(' ')
print('Ezf', Ezf)
print('Ezf prior:')
print(expit(Wf*Eh_mod+Uf*u+bf))
print(' ')
print('Ezp', Ezp)
print('Ezp prior:')
print(expit(Wp*Eh_mod+Up*u+bp))
print(' ')
print('Ezo', Ezo)
print('Ezo prior:')
print(expit(Wo*Eh_mod+Uo*u+bo))
print(' ')
print('E_omega_i:', Eomega_i)
print('E_omega_f:', Eomega_f)
print('E_omega_p:', Eomega_p)
print('E_omega_o:', Eomega_o)
print('gi', gi)
print('gf', gf)
print('gp', gp)
print('go', go)

print('elbo:', elbo)


plt.plot(np.arange(k), elbo_vec)
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

'''
