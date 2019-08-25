import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
from E_PG import qdf_log_pdf, entropy_q
from scipy import integrate

def generate(T, mu_0, covar_c, h_0, covar_h, Wi, Wf, Wp, Wo):
    c = np.zeros(T)
    h = np.zeros(T)
    v = np.zeros(T)
    zi = np.zeros(T)
    zf = np.zeros(T)
    zp = np.zeros(T)
    zo = np.zeros(T)

    zi[0] = np.random.binomial(1, expit(Wi*h_0))
    zf[0] = np.random.binomial(1, expit(Wf*h_0))
    zp[0] = np.random.binomial(1, expit(Wp*h_0))
    zo[0] = np.random.binomial(1, expit(Wo*h_0))

    c[0] = np.random.normal(zf[0]*mu_0+zi[0]*(2*zp[0]-1),
                            np.sqrt(covar_c[0,0]))
    v[0] = np.random.binomial(1,  expit(c[0]))
    h[0] = np.random.normal(zo[0]*(2*v[0]-1), np.sqrt(covar_h[0,0])) 
    for t in range(1,T):
        zi[t] = np.random.binomial(1, expit(Wi*h[t-1]))
        zf[t] = np.random.binomial(1, expit(Wf*h[t-1]))
        zp[t] = np.random.binomial(1, expit(Wp*h[t-1]))
        zo[t] = np.random.binomial(1, expit(Wo*h[t-1]))
        c[t] = np.random.normal(zf[t]*c[t-1]+zi[t]*(2*zp[t]-1),
                                np.sqrt(covar_c[t,t]))
        v[t] = np.random.binomial(1,  expit(c[t]))
        h[t] = np.random.normal(zo[t]*(2*v[t]-1), 
                                np.sqrt(covar_h[t,t])) 
    return c, h, v, zi, zf, zp, zo

####update c ########################################################
def update_qc(T, mu_0, covar, Ezi, Ezf, Ezp, Ev, E_gamma):
    Lambda = np.zeros((T,T))
    Lambda_m = np.zeros(T)

    ones = np.ones(T)
    
    #Construct Precision Diagonal
    diag1 = 1/np.diag(covar)
    diag2 = np.zeros(T)
    diag2[:-1] = diag1[1:]*Ezf[1:]
    Lambda = np.diag(diag1+diag2+4*E_gamma)

    #Construct Precision Off diagonals
    off_diag = -1/(np.diag(covar)[1:])*Ezf[1:]
    Lambda += np.diag(off_diag, k=1)
    Lambda += np.diag(off_diag, k=-1)

    #Construct Precision times mean
    Lambda_m = np.zeros(T)
    Lambda_m[0] = 1/covar[0,0]*mu_0*Ezf[0]
    Lambda_m += diag1*Ezi*(2*Ezp-ones)
    Lambda_m += 2*(Ev-1/2)*ones
    Lambda_m[:-1] += -diag1[1:]*Ezi[1:]*Ezf[1:]*(2*Ezp[1:]-ones[1:])
    
    return  Lambda, Lambda_m

def get_c_expects(Lambda, Lambda_m):
    #will need to update with message passing
    Sigma = np.linalg.inv(Lambda)
    m = Sigma @ Lambda_m
    Ecc = np.outer(m, m) + Sigma
    return m, Ecc

###############################################################


def update_qh(T, h_0, covar, Wi, Wf, Wp, Wo, Ev, Eomega_i, Eomega_f, 
              Eomega_p, Eomega_o, Ezi, Ezf, Ezp, Ezo):

    Lambda = np.zeros(T)
    Lambda_m = np.zeros(T)
    
    Lambda += 1/np.diag(covar)
    Lambda[:-1] += Eomega_i[1:]*Wi**2
    Lambda[:-1] += Eomega_f[1:]*Wf**2
    Lambda[:-1] += Eomega_p[1:]*Wp**2
    Lambda[:-1] += Eomega_o[1:]*Wo**2
    

    Lambda_m += 1/np.diag(covar)*Ezo*(2*Ev-1)
    
    Lambda_m[:-1] += (Ezi[1:]-1/2)*Wi
    Lambda_m[:-1] += (Ezf[1:]-1/2)*Wf
    Lambda_m[:-1] += (Ezp[1:]-1/2)*Wp
    Lambda_m[:-1] += (Ezo[1:]-1/2)*Wo
    
    return Lambda, Lambda_m
    
def get_h_expects(Lambda, Lambda_m):
    Sigma = 1/Lambda
    m = Sigma*Lambda_m
    Ehh = m**2 + Sigma
    return m, Ehh
    
#############################################################
def update_q_gamma(Ecc):
    g = np.sqrt(4*np.diag(Ecc))
    b = np.ones(len(g))
    E_gamma = b/(2*g)*np.tanh(g/2)
    return b, g, E_gamma


def update_q_omega_star(Ehh, h_0, W_star):
    g = np.zeros(len(Ehh))
    g[0] = W_star*h_0
    g[1:] = np.sqrt(W_star**2*Ehh[:-1])
    b = np.ones(len(g))
    E_omega = b/(2*g)*np.tanh(g/2)
    return b, g, E_omega
##############################################################

def update_qv(Eh, covar, Ec, Ezo):
    diag = np.diag(covar)
    value = 2/diag*Eh*Ezo+2*Ec
    return expit(value)
############################################################

def W_Eh_z_update(W_star, Eh, h_0, value):
    value[0] += W_star*h_0
    value[1:] += Eh[:-1]
    return value

def update_zi(mu_0, covar, Wi, h_0, Eh, Ec, Ezp, Ezf):
    diag = np.diag(covar)
    ones = np.ones(len(Ec))
    
    value = 1/diag*Ec*(2*Ezp-ones)
    
    Ec_minus = np.zeros(len(Ec))
    Ec_minus[0] = mu_0
    Ec_minus[1:] = Ec[:-1]
    value += -1/diag*Ezf*Ec_minus*(2*Ezp-1)-(1/2)*(1/diag)
    
    value = W_Eh_z_update(Wi, Eh, h_0, value)
    
    return expit(value)

def update_zf(mu_0, covar, Wf, h_0, Eh,  Ec, Ecc, Ezi, Ezp):
    ones = np.ones(len(Ec))
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

    value += -1/diag*Ec_min1*Ezi*(2*Ezp-ones)
    
    value = W_Eh_z_update(Wf, Eh, h_0, value)
    
    return expit(value)

def update_zp(mu_0, covar, Wp, h_0, Eh, Ec, Ezi, Ezf):
    diag = np.diag(covar)
    
    value = 2/diag*Ec*Ezi

    Ec_minus = np.zeros(len(Ec))
    Ec_minus[0] = mu_0
    Ec_minus[1:] = Ec[:-1]

    value += -2/diag*Ezf*Ec_minus*Ezi

    value = W_Eh_z_update(Wp, Eh, h_0, value)
    return expit(value)

def update_zo(h_0, covar, Wo, Eh, Ev):
    diag = np.diag(covar)
    
    value = 1/diag*Eh*(2*Ev-1)-1/2*1/diag

    value = W_Eh_z_update(Wo, Eh, h_0, value)
    return expit(value)

                    

###########################################################
##ELBO calculation###


def elbo_c(mu_0, m, covar, Ecc):
    ones = np.ones(len(m))
    diag = np.diag(covar)
    Ecc_diag = np.diag(Ecc)

    value = np.log( 1/np.sqrt(2*np.pi*diag) )
    value += -1/2*( 1/diag )*Ecc_diag

    term = 1/diag
    term[0] *= m[0]*mu_0*Ezf[0]
    term[1:] *= np.diag(Ecc,-1)*Ezf[1:]
    value += term

    value += 1/diag*Ezi*(2*Ezp-ones)*m
    
    term = -1/2*1/diag
    term[0] *= mu_0**2*Ezf[0]
    term[1:] *= np.diag(Ecc)[:-1]*Ezf[1:]
    value += term

    term = -1/diag*Ezi*Ezf*(2*Ezp-ones)
    term[0] *= mu_0
    term[1:] *= m[:-1]
    value += term
    
    value += -1/2*1/diag*Ezi

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

    
def elbo_z_star(Ez,Eh, h_0, W_star, star):
    ones = np.ones(len(Eh))
    
    mod_Eh = np.zeros(len(Eh))
    mod_Eh[0] = h_0
    mod_Eh[1:] = Eh[:-1]
    value = np.log(1/2)*ones + (Ez-1/2*ones)*W_star*mod_Eh
    print('elbo_{}'.format(star), np.sum(value))
    return np.sum(value)


def elbo_v(Ev, Ec):
    ones = np.ones(len(Ev))
    value = np.log(1/2)*ones + 2*(Ev-1/2*ones)*Ec
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
                           epsabs=1e-4, epsrel = 0)[0]
    print('elbo_gamma', value)
    return value

def elbo_omega_star_1(Eomega_star, Ehh, h_0, W_star, star):
    mod_Ehh = np.zeros(len(Ehh))
    mod_Ehh[0] = h_0**2
    mod_Ehh[1:] = Ehh[:-1]
    value = -1/2*Eomega_star*(W_star**2)*mod_Ehh
    print('elbo_omega1_{}:'.format(star), np.sum( value) )
    return np.sum( value )

def elbo_omega2_star(Ehh, h_0, W_star, star):
    mod_Ehh = np.zeros(len(Ehh))
    mod_Ehh[0] = h_0**2
    mod_Ehh[1:] = Ehh[:-1]
    value = 0
    for t in range(0,len(mod_Ehh)):
        value += integrate.quad(qdf_log_pdf, 0, np.inf,
                           args=(1,0, np.sqrt(W_star**2*mod_Ehh[t])),
                           epsabs=1e-4, epsrel = 0)[0]
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
                                epsabs=1e-4, epsrel=0)[0]
    print('gam_entrpy:', value)
    return value

def entropy_omega_star(Ehh, h_0, W_star, star):
    mod_Ehh = np.zeros(len(Ehh))
    mod_Ehh[0] = h_0**2
    mod_Ehh[1:] = Ehh[:-1]
    value = 0
    for t in range(0,len(Ehh)):
        value += integrate.quad(entropy_q, 0, np.inf, 
                                args=(1,np.sqrt(W_star**2*mod_Ehh[t])),
                                epsabs=1e-4, epsrel=0)[0]
    print('omega_{}_entrpy:'.format(star), value)
    return value

def get_elbo(T,  mu_0, covar_c, h_0, covar_h, Wi, Wf, Wp, Wo, 
             Ec, Ecc, Ev, Eh, Ehh, Ezi, Ezf, Ezp, Ezo, E_gamma, 
             Eomega_i, Eomega_f, Eomega_p, Eomega_o):
    #NOTE: currently excludes several PG terms
    elbo = elbo_c(mu_0, Ec, covar_c, Ecc)
    elbo += elbo_v(Ev, Ec)
    elbo += elbo_h(h_0, covar_h, Eh, Ehh, Ezo, Ev)
    
    elbo += elbo_z_star(Ezi,Eh, h_0, Wi, 'zi')
    elbo += elbo_z_star(Ezf,Eh, h_0, Wf, 'zf')
    elbo += elbo_z_star(Ezp,Eh, h_0, Wp, 'zp')
    elbo += elbo_z_star(Ezo,Eh, h_0, Wo, 'zo')
    
    elbo += elbo_gamma1(E_gamma, Ecc)
    elbo += elbo_gamma2(Ecc)

    elbo += elbo_omega_star_1(Eomega_i, Ehh, h_0, Wi, 'i')
    elbo += elbo_omega_star_1(Eomega_f, Ehh, h_0, Wf, 'f')
    elbo += elbo_omega_star_1(Eomega_p, Ehh, h_0, Wp, 'p')
    elbo += elbo_omega_star_1(Eomega_o, Ehh, h_0, Wo, 'o')

    ###
    #multiply times 4 since Wi=Wf=Wp input to save time
    elbo_omega2 = elbo_omega2_star(Ehh, h_0, Wi, 'i')
    elbo += elbo_omega2*4
    '''
    elbo += elbo_omega2_star(Ehh, h_0, Wi, 'i')
    elbo += elbo_omega2_star(Ehh, h_0, Wf, 'f')
    elbo += elbo_omega2_star(Ehh, h_0, Wp, 'p')
    elbo += elbo_omega2_star(Ehh, h_0, Wo, 'o')
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
    ent_omega = entropy_omega_star(Ehh, h_0, Wi, 'i')
    elbo += ent_omega*4
    '''
    elbo += entropy_omega_star(Ehh, h_0, Wi, 'i')
    elbo += entropy_omega_star(Ehh, h_0, Wf, 'f')
    elbo += entropy_omega_star(Ehh, h_0, Wp, 'p')
    elbo += entropy_omega_star(Ehh, h_0, Wo, 'o')
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
T=3


covar_c = np.identity(T)*.3*np.ones(T)
mu_0 = .4

covar_h = np.identity(T)*.4*np.ones(T)
h_0 = .6

Wi = 1
Wf = 1
Wp = 1   
Wo = 1
     
c,h,v,zi,zf,zp,zo = generate(T, mu_0, covar_c, h_0, 
                          covar_h, Wi, Wf, Wp, Wo)

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


#Initialize
E_gamma = .3*np.ones(T)
Ev = .8*np.ones(T)
Ezi = .3*np.ones(T)
Ezf = .3*np.ones(T)
Ezp = .3*np.ones(T)
Ezo = .3*np.ones(T)
Eomega_i = .3*np.ones(T)
Eomega_f = .3*np.ones(T)
Eomega_p = .3*np.ones(T)
Eomega_o = .3*np.ones(T)


Ec_old = np.ones(T)*np.inf
Ecc_old = np.ones((T,T))*np.inf
Eh_old = np.ones(T)*np.inf
Ehh_old = np.ones((T,T))*np.inf
Ev_old = np.ones(T)*np.inf
g_old = np.ones(T)*np.inf
Ezi_old = np.ones(T)*np.inf
Ezf_old = np.ones(T)*np.inf
Ezp_old = np.ones(T)*np.inf
Ezo_old = np.ones(T)*np.inf
gi_old = np.ones(T)*np.inf
gf_old = np.ones(T)*np.inf
gp_old = np.ones(T)*np.inf
go_old = np.ones(T)*np.inf

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
    Lambda_h, Lambda_h_m = update_qh(T, h_0, covar_h, 
                                     Wi, Wf, Wp, Wo, Ev, 
                                     Eomega_i, Eomega_f, Eomega_p, 
                                     Eomega_o, Ezi, Ezf, Ezp, Ezo)
    Eh, Ehh = get_h_expects(Lambda_h, Lambda_h_m)

    diff_list, Eh_old = get_diff( Eh, Eh_old, diff_list)
    diff_list, Ehh_old = get_diff( Ehh, Ehh_old, diff_list)

    #update qv
    Ev = update_qv(Eh, covar_h, Ec, Ezo)
    diff_list, Ev_old = get_diff( Ev, Ev_old, diff_list)
        
    #update q_gamma
    b, g, E_gamma = update_q_gamma(Ecc)
    diff_list, g_old = get_diff( g, g_old, diff_list)
    

    #update q_omegas
    bi, gi, Eomega_i = update_q_omega_star(Ehh, h_0, Wi)
    diff_list, gi_old = get_diff( gi, gi_old, diff_list)


    bf, gf, Eomega_f = update_q_omega_star(Ehh, h_0, Wf)
    diff_list, gf_old = get_diff( gf, gf_old, diff_list)

    bp, gp, Eomega_p = update_q_omega_star(Ehh, h_0, Wp)
    diff_list, gp_old = get_diff( gp, gp_old, diff_list)

    bo, go, Eomega_o = update_q_omega_star(Ehh, h_0, Wo)
    diff_list, go_old = get_diff( go, go_old, diff_list)
    
    #update q_zi
    Ezi = update_zi(mu_0, covar_c, Wi, h_0, Eh, Ec, Ezp, Ezf)
    diff_list, Ezi_old = get_diff( Ezi, Ezi_old, diff_list)

    #update q_zf
    Ezf = update_zf(mu_0, covar_c, Wf, h_0, Eh,  Ec, Ecc, Ezi, Ezp)
    diff_list, Ezf_old = get_diff( Ezf, Ezf_old, diff_list)

    #update q_zp
    Ezp = update_zp(mu_0, covar_c, Wp, h_0, Eh, Ec, Ezi, Ezf)
    diff_list, Ezp_old = get_diff( Ezp, Ezp_old, diff_list)
    
    #update q_zo
    Ezo = update_zo(h_0, covar_h, Wo, Eh, Ev)
    diff_list, Ezo_old = get_diff( Ezo, Ezo_old, diff_list)


    #convergence check
    

    diff = np.amax( diff_list )
    diff_vec.append(diff)
    
    '''
    elbo = get_elbo(T,  mu_0, covar_c, h_0, covar_h, Wi, Wf, Wp, Wo, 
             Ec, Ecc, Ev, Eh, Ehh, Ezi, Ezf, Ezp, Ezo, E_gamma, 
             Eomega_i, Eomega_f, Eomega_p, Eomega_o)

    elbo_vec.append(elbo)
    print(' ')
    '''
    k+=1


elbo = get_elbo(T,  mu_0, covar_c, h_0, covar_h, Wi, Wf, Wp, Wo, 
             Ec, Ecc, Ev, Eh, Ehh, Ezi, Ezf, Ezp, Ezo, E_gamma, 
             Eomega_i, Eomega_f, Eomega_p, Eomega_o)

elbo_vec.append(elbo)
print(' ')



print('Ec:', Ec)
Sigma = Ecc-np.outer(Ec,Ec)
print('Sigma:')
print(Sigma)
print('Eh:',Eh) 
print('Ehh', Ehh)
Sigma_h = Ehh-Eh**2
print('Sigma_h:', Sigma_h)
print('Ev:', Ev)
print('E_gamma:', E_gamma)
print('g:', g)  
print('Ezi', Ezi)
print('Ezf', Ezf)
print('Ezp', Ezp)
print('Ezo', Ezo)
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

