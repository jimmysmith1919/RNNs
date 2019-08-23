import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
from E_PG import qdf_log_pdf, entropy_q
from scipy import integrate

def generate(T, mu_0, pi, pf, pp, covar):
    c = np.zeros(T)
    v = np.zeros(T)
    
    zi = np.random.binomial(1, expit(pi), size=T)
    zf = np.random.binomial(1, expit(pf), size=T)
    zp = np.random.binomial(1, expit(pp), size=T)


    c = np.random.normal(zf*mu_0+zi*(2*zp-1),
                            np.sqrt(covar))
    v = v_gen = np.random.binomial(1,  expit(c))
    
    return c, v, zi, zf, zp

####update c ########################################################
def update_qc(T, mu_0, covar, Ezi, Ezf, Ezp, Ev, E_gamma):
    Lambda = 1/covar+E_gamma
    Lambda_m = 1/covar*Ezf*mu_0
    Lambda_m += 1/covar*Ezi*(2*Ezp-1)
    Lambda_m += Ev-1/2
    
    return  Lambda, Lambda_m

def get_c_expects(Lambda, Lambda_m):
    #will need to update with message passing
    Sigma = 1/Lambda
    m = Sigma*Lambda_m
    Ecc = m**2 + Sigma
    return m, Ecc

###############################################################

def update_q_gamma(Ecc):
    g = np.sqrt(Ecc)
    b = 1
    E_gamma = b/(2*g)*np.tanh(g/2)
    return b, g, E_gamma
##############################################################

def update_qv(Ec): 
    return expit(Ec)
############################################################

def update_zi(mu_0, covar, Ec, Ezp, Ezf, pi):
    
    value = 1/covar*Ec*(2*Ezp-1)
    value += -1/covar*Ezf*mu_0*(2*Ezp-1)
    value += -1/2*1/covar
    value += np.log( pi/(1-pi) )    ###will need to fix
    
    return expit(value)

def update_zf(mu_0, covar, Ec, Ecc, Ezi, Ezp, pf):
    
    
    value = 1/covar*Ec*mu_0
    value += -1/2*1/covar*mu_0**2
    value += -1/covar*mu_0*Ezi*(2*Ezp-1)
    value += np.log( pf/(1-pf)  )
    return expit(value)

def update_zp(mu_0, covar, Ec, Ezi, Ezf, pp):
    value = 2/covar*Ec*Ezi
    value += -2/covar*Ezf*mu_0*Ezi
    value += np.log( pp/(1-pp)  )
    return expit(value)
                    

###########################################################
##ELBO calculation###


def elbo_c(mu_0, m, covar, Ecc):
    value = np.log( 1/np.sqrt(2*np.pi*covar) )
    value += -1/2*( 1/covar )*Ecc
    value += 1/covar*m*mu_0*Ezf
    value += 1/covar*Ezi*(2*Ezp-1)*m
    
    value += -1/2*1/covar*mu_0**2*Ezf
    value += -1/covar*Ezi*Ezf*(2*Ezp-1)*mu_0
    value += -1/2*1/covar*Ezi

    print('elbo_c:', np.sum(value))
    return np.sum(value)

def elbo_z_star(Ez, p, star):
    value = Ez*np.log(p)+(1-Ez)*np.log(1-p)
    print('elbo_{}'.format(star), np.sum(value))
    return np.sum(value)
    
def elbo_v(Ev, m):
    value = np.log(1/2) + (Ev-1/2)*m
    print('elbo_v:', np.sum(value))
    return np.sum(value)

def elbo_gamma1(E_gamma, Ecc):
    value = -1/2*E_gamma*Ecc
    print('elbo_gamma1:', np.sum( value) )
    return np.sum( value )

def elbo_gamma2(Ecc):
    Ecc_diag = Ecc
    value = 0
    value += integrate.quad(qdf_log_pdf, 0, np.inf,
                           args=(1,0, np.sqrt(Ecc_diag)),
                           epsabs=1e-3, epsrel = 0)[0]
        
    print('elbo_gamma', value)
    return value


def entropy_c(T, m, Ecc):
    #need to make more efficient
    Sigma = Ecc-m**2
    print('Entropy_c:', T/2*(1+np.log(2*np.pi))+
          1/2*np.log( Sigma ))
    return T/2*(1+np.log(2*np.pi))+1/2*np.log( Sigma )

def entropy_Bern(p, str):
    if np.isclose(p,0) or np.isclose(p,1):
        value = 0
    else:
        value = -p*np.log(p)-(1-p)*np.log(1-p)
    print('entropy_{}:'.format(str), np.sum(value))
    return np.sum(value)


def entropy_gamma(Ecc):
    Ecc_diag = Ecc
    value = 0
    value += integrate.quad(entropy_q, 0, np.inf, 
                                args=(1,np.sqrt(Ecc_diag)),
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

np.random.seed(0)

T=1

#var = np.random.uniform(.1, .5, size=T)
covar = .3#np.identity(T)*.3*np.ones(T)#var
mu_0 =  30
pi = .6 
pf = .3
pp = .4

        
c,v, zi, zf, zp = generate(T, mu_0, pi, pf, pp, covar)

print('Mu_0=',mu_0)
print('covar=')
print(covar)
print('c:', c)
print('Sigmoid(c):', expit(c))
print('v:', v)
print('zi',zi)
print('zf', zf)
print('zp', zp)
print(' ')


#Initialize q_omega
E_gamma = .3
Ev = .8
Ezi = .3
Ezf = .3
Ezp = .3

m_old = np.ones(T)*np.inf
Ecc_old = np.ones((T,T))*np.inf
Ev_old = np.ones(T)*np.inf
g_old = np.ones(T)*np.inf
Ezi_old = np.ones(T)*np.inf
Ezf_old = np.ones(T)*np.inf
Ezp_old = np.ones(T)*np.inf


diff = np.inf
tol = .001

diff_vec = []
elbo_vec = []

k = 0

while diff > tol:
    
    #update qc
    Lambda, Lambda_m = update_qc(T, mu_0, covar, Ezi, Ezf, Ezp, Ev, E_gamma)
    m, Ecc = get_c_expects(Lambda, Lambda_m) 

    #update qv
    Ev = update_qv(m)
        
    #update q_omega
    b, g, E_gamma = update_q_gamma(Ecc)
    
    #update q_zi
    Ezi = update_zi(mu_0, covar, m, Ezp, Ezf, pi)

    #update q_zf
    Ezf = update_zf(mu_0, covar, m, Ecc, Ezi, Ezp, pf)

    #update q_zp
    Ezp = update_zp(mu_0, covar, m, Ezi, Ezf, pp)

    #convergence check
    m_diff = np.amax(np.absolute(m-m_old))
    m_old = m
    
    Ecc_diff = np.amax(np.absolute(Ecc-Ecc_old))
    Ecc_old = Ecc

    Ev_diff = np.amax(np.absolute(Ev-Ev_old))
    Ev_old = Ev

    g_diff = np.amax(np.absolute(g-g_old))
    g_old = g

    Ezi_diff = np.amax(np.absolute(Ezi-Ezi_old))
    Ezi_old = Ezi

    Ezf_diff = np.amax(np.absolute(Ezf-Ezf_old))
    Ezf_old = Ezf

    Ezp_diff = np.amax(np.absolute(Ezp-Ezp_old))
    Ezp_old = Ezp

    diff = np.amax( [m_diff, Ecc_diff, Ev_diff, g_diff, 
                     Ezi_diff, Ezf_diff, Ezp_diff] )
    diff_vec.append(diff)

    elbo = get_elbo(T,  mu_0, covar, m, Ecc, Ev, 
             Ezi, Ezf, Ezp, E_gamma, pi, pf, pp)
    elbo_vec.append(elbo)

    k+=1


print('m:', m)
Sigma = Ecc-np.outer(m,m)
print('Sigma:')
print(Sigma)
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

