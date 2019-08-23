import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
from E_PG import qdf_log_pdf, entropy_q
from scipy import integrate

def generate(T, mu_0, covar_c, h_0, covar_h, Wi, Wf, Wp, alpha):
    c = np.zeros(T)
    h = np.zeros(T)
    h = np.random.normal(alpha, np.sqrt(np.diag(covar_h)), size = T)
    v = np.zeros(T)
    zi = np.zeros(T)
    zf = np.zeros(T)
    zp = np.zeros(T)
    

    zi[0] = np.random.binomial(1, expit(Wi*h_0))
    zf[0] = np.random.binomial(1, expit(Wf*h_0))
    zp[0] = np.random.binomial(1, expit(Wp*h_0))

    c[0] = np.random.normal(zf[0]*mu_0+zi[0]*(2*zp[0]-1),
                            np.sqrt(covar_c[0,0]))
    v[0] = v_gen = np.random.binomial(1,  expit(c[0]))
    for t in range(1,T):
        zi[t] = np.random.binomial(1, expit(Wi*h[t-1]))
        zf[t] = np.random.binomial(1, expit(Wf*h[t-1]))
        zp[t] = np.random.binomial(1, expit(Wp*h[t-1]))
        c[t] = np.random.normal(zf[t]*c[t-1]+zi[t]*(2*zp[t]-1),
                                np.sqrt(covar_c[t,t]))
        v[t] = np.random.binomial(1,  expit(c[t]))
    
    return c, h, v, zi, zf, zp

####update c ########################################################
def update_qc(T, mu_0, covar, Ezi, Ezf, Ezp, Ev, E_gamma):
    Lambda = np.zeros((T,T))
    Lambda_m = np.zeros(T)

    ones = np.ones(T)
    
    #Construct Precision Diagonal
    diag1 = 1/np.diag(covar)
    diag2 = np.zeros(T)
    diag2[:-1] = diag1[1:]*Ezf[1:]
    Lambda = np.diag(diag1+diag2+E_gamma)

    #Construct Precision Off diagonals
    off_diag = -1/(np.diag(covar)[1:])*Ezf[1:]
    Lambda += np.diag(off_diag, k=1)
    Lambda += np.diag(off_diag, k=-1)

    #Construct Precision times mean
    Lambda_m = np.zeros(T)
    Lambda_m[0] = 1/covar[0,0]*mu_0*Ezf[0]
    Lambda_m += diag1*Ezi*(2*Ezp-ones)
    Lambda_m += Ev-1/2*ones
    Lambda_m[:-1] += -diag1[1:]*Ezi[1:]*Ezf[1:]*(2*Ezp[1:]-ones[1:])
    
    return  Lambda, Lambda_m

def get_c_expects(Lambda, Lambda_m):
    #will need to update with message passing
    Sigma = np.linalg.inv(Lambda)
    m = Sigma @ Lambda_m
    Ecc = np.outer(m, m) + Sigma
    return m, Ecc

###############################################################


def update_qh(T, h_0, covar, Wi, Wf, Wp,  Eomega_i, Eomega_f, 
              Eomega_p, Ezi, Ezf, Ezp, alpha):

    Lambda = np.zeros(T)
    Lambda_m = np.zeros(T)
    
    Lambda += 1/np.diag(covar)
    Lambda[:-1] += Eomega_i[1:]*Wi**2
    Lambda[:-1] += Eomega_f[1:]*Wf**2
    Lambda[:-1] += Eomega_p[1:]*Wp**2
    

    Lambda_m += 1/np.diag(covar)*alpha
    Lambda_m[:-1] += (Ezi[1:]-1/2)*Wi
    Lambda_m[:-1] += (Ezf[1:]-1/2)*Wf
    Lambda_m[:-1] += (Ezp[1:]-1/2)*Wp
    
    return Lambda, Lambda_m
    
def get_h_expects(Lambda, Lambda_m):
    Sigma = 1/Lambda
    m = Sigma*Lambda_m
    Ehh = m**2 + Sigma
    return m, Ehh
    
#############################################################
def update_q_gamma(Ecc):
    g = np.sqrt(np.diag(Ecc))
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

def update_qv(Ec): 
    return expit(Ec)
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

def elbo_z_star(Ez, p, star):
    ones= np.ones(len(p))
    value = Ez*np.log(p)+(ones-Ez)*np.log(ones-p)
    print('elbo_{}'.format(star), np.sum(value))
    return np.sum(value)
    


def elbo_v(Ev, m):
    ones = np.ones(len(Ev))
    value = np.log(1/2)*ones + (Ev-1/2*ones)*m
    print('elbo_v:', np.sum(value))
    return np.sum(value)

def elbo_gamma1(E_gamma, Ecc):
    value = -1/2*E_gamma*np.diag(Ecc)
    print('elbo_gamma1:', np.sum( value) )
    return np.sum( value )

def elbo_gamma2(Ecc):
    Ecc_diag = np.diag(Ecc)
    value = 0
    for t in range(0,len(Ecc_diag)):
        value += integrate.quad(qdf_log_pdf, 0, np.inf,
                           args=(1,0, np.sqrt(Ecc_diag[t])),
                           epsabs=1e-3, epsrel = 0)[0]
    print('elbo_gamma', value)
    return value


def entropy_c(T, m, Ecc):
    #need to make more efficient
    Sigma = Ecc-np.outer(m,m)
    print('Entropy_c:', T/2*(1+np.log(2*np.pi))+
          1/2*np.log( np.linalg.det(Sigma) ))
    return T/2*(1+np.log(2*np.pi))+1/2*np.log( np.linalg.det(Sigma) )

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
                                args=(1,np.sqrt(Ecc_diag[t])),
                                epsabs=1e-3, epsrel=0)[0]
    print('gam_entrpy:', value)
    print(' ')
    return value

def get_elbo(T,  mu_0, covar, m, Ecc, Ev, 
             Ezi, Ezf, Ezp, E_gamma, pi, pf, pp):
    #NOTE: currently excludes several PG terms
    elbo = elbo_c(mu_0, m, covar, Ecc)
    elbo += elbo_v(Ev, m)
    elbo += elbo_z_star(Ezi, pi, 'zi')
    elbo += elbo_z_star(Ezf, pf, 'zf')
    elbo += elbo_z_star(Ezp, pp, 'zp')
    elbo += elbo_gamma1(E_gamma, Ecc)
    elbo += elbo_gamma2(Ecc)
    elbo += entropy_c(T, m, Ecc)
    elbo += entropy_Bern(Ev, 'v')
    elbo += entropy_Bern(Ezi, 'zi')
    elbo += entropy_Bern(Ezf, 'zf')
    elbo += entropy_Bern(Ezp, 'zp')
    elbo += entropy_gamma(Ecc)
    return elbo


#####
np.random.seed(0)
T=3


covar_c = np.identity(T)*.3*np.ones(T)
mu_0 = .7

covar_h = np.identity(T)*.3*np.ones(T)
h_0 = .3
alpha = np.ones(T)*h_0

Wi = 1
Wf = 1
Wp = 1   
     
c,h,v,zi,zf,zp = generate(T, mu_0, covar_c, h_0, 
                          covar_h, Wi, Wf, Wp, alpha)

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
print(' ')


#Initialize
E_gamma = .3*np.ones(T)
Ev = .8*np.ones(T)
Ezi = .3*np.ones(T)
Ezf = .3*np.ones(T)
Ezp = .3*np.ones(T)
Eomega_i = .3*np.ones(T)
Eomega_f = .3*np.ones(T)
Eomega_p = .3*np.ones(T)


Ec_old = np.ones(T)*np.inf
Ecc_old = np.ones((T,T))*np.inf
Eh_old = np.ones(T)*np.inf
Ehh_old = np.ones((T,T))*np.inf
Ev_old = np.ones(T)*np.inf
g_old = np.ones(T)*np.inf
Ezi_old = np.ones(T)*np.inf
Ezf_old = np.ones(T)*np.inf
Ezp_old = np.ones(T)*np.inf
gi_old = np.ones(T)*np.inf
gf_old = np.ones(T)*np.inf
gp_old = np.ones(T)*np.inf

diff = np.inf
tol = .001


diff_vec = []
elbo_vec = []

k = 0


def get_diff(param, param_old, diff_list):
    param_diff = np.amax(np.absolute(param-param_old))
    diff_list.append(param_diff)
    param_old = param
    return diff_list, param_old

while diff > tol:
    diff_list = []

    #update qc
    Lambda_c, Lambda_c_m = update_qc(T, mu_0, covar_c, 
                                     Ezi, Ezf, Ezp, Ev, E_gamma)
    Ec, Ecc = get_c_expects(Lambda_c, Lambda_c_m) 
    diff_list, Ec_old = get_diff( Ec, Ec_old, diff_list)

    diff_list, Ecc_old = get_diff( Ecc, Ecc_old, diff_list)
    
    #update qv
    Ev = update_qv(Ec)
    diff_list, Ev_old = get_diff( Ev, Ev_old, diff_list)
        
    #update q_gamma
    b, g, E_gamma = update_q_gamma(Ecc)
    diff_list, g_old = get_diff( g, g_old, diff_list)
    
    #update q_h:
    Lambda_h, Lambda_h_m = update_qh(T, h_0, covar_h, Wi, Wf, Wp, 
                                     Eomega_i, Eomega_f, Eomega_p, 
                                     Ezi, Ezf, Ezp, alpha)
    Eh, Ehh = get_h_expects(Lambda_h, Lambda_h_m)

    diff_list, Eh_old = get_diff( Eh, Eh_old, diff_list)
    diff_list, Ehh_old = get_diff( Ehh, Ehh_old, diff_list)

    #update q_omegas
    bi, gi, Eomega_i = update_q_omega_star(Ehh, h_0, Wi)
    diff_list, gi_old = get_diff( gi, gi_old, diff_list)


    bf, gf, Eomega_f = update_q_omega_star(Ehh, h_0, Wf)
    diff_list, gf_old = get_diff( gf, gf_old, diff_list)

    bp, gp, Eomega_p = update_q_omega_star(Ehh, h_0, Wp)
    diff_list, gp_old = get_diff( gp, gp_old, diff_list)
    
    #update q_zi
    Ezi = update_zi(mu_0, covar_c, Wi, h_0, Eh, Ec, Ezp, Ezf)
    diff_list, Ezi_old = get_diff( Ezi, Ezi_old, diff_list)

    #update q_zf
    Ezf = update_zf(mu_0, covar_c, Wf, h_0, Eh,  Ec, Ecc, Ezi, Ezp)
    diff_list, Ezf_old = get_diff( Ezf, Ezf_old, diff_list)

    #update q_zp
    Ezp = update_zp(mu_0, covar_c, Wp, h_0, Eh, Ec, Ezi, Ezf)
    diff_list, Ezp_old = get_diff( Ezp, Ezp_old, diff_list)
    
    #convergence check
    

    diff = np.amax( diff_list )
    diff_vec.append(diff)

    #elbo = get_elbo(T,  mu_0, covar, Ec, Ecc, Ev, 
    #         Ezi, Ezf, Ezp, E_gamma, pi, pf, pp)
    #elbo_vec.append(elbo)

    k+=1


print('Ec:', Ec)
Sigma = Ecc-np.outer(Ec,Ec)
print('Sigma:')
print(Sigma)
print('Eh:',Eh) 
print('Ev:', Ev)
print('g:', g)  
print('Ezi', Ezi)
print('Ezf', Ezf)
print('Ezp', Ezp)

print('elbo:', elbo)


plt.plot(np.arange(k), elbo_vec)
plt.xlabel('Iteration')
plt.ylabel('ELBO')
plt.title('ELBO convergence (excludes some PG terms)')
plt.savefig('ELBO_0.png')
plt.show()
plt.close()

plt.plot(np.arange(k), diff_vec)
plt.xlabel('Iteration')
plt.ylabel('Max Parameter Difference')
plt.savefig('Error.png')
plt.show()
plt.close()

