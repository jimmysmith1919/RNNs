import numpy as np
from scipy.special import expit
from scipy import integrate
import matplotlib.pyplot as plt
from E_PG import qdf_log_pdf, entropy_q

def generate(mu_0, covar):
    c = np.random.normal(mu_0, np.sqrt(covar))
    v = np.random.binomial(1,  expit(c))
    return c, v

##update c ##

def update_diag(covar, E_gamma):
    return  1/covar+E_gamma
    
def update_L_m0(mu_0, covar, Ev, Ezi, Ezf, Ezp):
    #takes initial condition mu_0 into account
    return (Ev-1/2) + 1/covar*mu_0*Ezf+1/covar*Ezi*(2*Ezp-1)

def update_qc(T, mu_0, covar, Ev, E_gamma, Ezi, Ezf, Ezp):

    #c_0
    Lambda = update_diag(covar, E_gamma)
    Lambda_m = update_L_m0(mu_0, covar, Ev, Ezi, Ezf, Ezp)
    
    return  Lambda, Lambda_m

def get_c_expects(Lambda, Lambda_m):
    #will need to update with message passing
    Sigma = 1/Lambda
    m = Sigma * Lambda_m
    Ecc = m**2 + Sigma
    return m, Ecc

#######

def update_q_gamma(Ecc):
    g = np.sqrt(Ecc)
    b = 1
    E_gamma = b/(2*g)*np.tanh(g/2)
    return b, g, E_gamma


def update_qv(Ec): 
    return expit(Ec)

def update_qzi(mu_0, covar, Ec, Ezp, Ezf, pi):
    value = 1/covar*(2*Ezp-1)*Ec
    value += -1/covar*Ezf*mu_0*(2*Ezp-1)
    value += -1/2*1/covar
    value += np.log(pi/(1-pi))
    return expit(value)

def update_qzf(mu_0, covar, Ec, Ezi, Ezp, pf):
    value = 1/covar*mu_0*Ec
    value += -1/2*1/covar*mu_0**2
    value += -1/covar*Ezi*mu_0*(2*Ezp-1)
    value += np.log(pf/(1-pf))
    return expit(value)

def update_qzp(mu_0, covar, Ec, Ezi, Ezf, pp):
    value = 2/covar*Ezi*Ec
    value += -2/covar*Ezf*Ezi*mu_0
    value += np.log(pp/(1-pp))
    return expit(value)

##ELBO calculation###
def elbo_c(mu_0, m, covar, Ecc):
    value = np.log( 1/np.sqrt(2*np.pi*covar) ) 
    value +=  -1/2*( 1/covar )*Ecc 
    value += 1/covar*mu_0*m
    value += -1/2*1/covar*mu_0**2 
    print('elbo_c:', value)
    return value

def elbo_v(Ev, m):
    value = np.log(1/2) + (Ev-1/2)*m
    print('elbo_v:', value)
    return value

def elbo_gamma1(E_gamma, Ecc):
    print('elbo_gamma1:', -1/2*E_gamma*Ecc )
    return -1/2*E_gamma*Ecc 

def elbo_gamma2(Ecc):
    value = integrate.quad(qdf_log_pdf, 0, np.inf, 
                           args=(1,0, np.sqrt(Ecc)),
                           epsabs=1e-1, epsrel = 0)[0]
    print('elbo_gamma', value)
    return value

def entropy_c(T, m, Ecc):
    #need to make more efficient
    Sigma = Ecc-m**2
    print('Entropy_c:', 1/2*(1+np.log(2*np.pi))+1/2*np.log( Sigma ) )
    return 1/2*(1+np.log(2*np.pi))+1/2*np.log( Sigma )

def entropy_v(Ev):
    value = -Ev*np.log(Ev)-(1-Ev)*np.log(1-Ev)
    print('entropy_v:', value)
    return value

def entropy_gamma(Ecc):
    value = integrate.quad(entropy_q, 0, np.inf, args=(1,np.sqrt(Ecc)), 
                   epsabs=1e-1, epsrel=0)[0]
    print('gam_entrpy:', value)
    print(' ')
    return value




def get_elbo(T,  mu_0, covar, m, Ecc, Ev, E_gamma):
    #NOTE: currently excludes several PG terms
    elbo = elbo_c(mu_0, m, covar, Ecc)
    elbo += elbo_v(Ev, m)
    elbo += elbo_gamma1(E_gamma, Ecc)
    #elbo += elbo_gamma2(E_gamma)
    elbo += elbo_gamma2(Ecc)
    elbo += entropy_c(T, m, Ecc)
    elbo += entropy_v(Ev)
    #elbo += entropy_gamma(Ecc)
    elbo += entropy_gamma(Ecc)
    return elbo

np.random.seed(0)


T=1


covar = .3 
mu_0 = 30
pi = .8
pf = .5
pp = .5
        
c,v = generate(mu_0, covar)

print('Mu_0=',mu_0)
print('covar=')
print(covar)
print('c:', c)
print('Sigmoid(c):', expit(c))
print('v:', v)
print(' ')


#Initialize q_omega
E_gamma = .5 
Ev = .5 
Ezi = .3
Ezf = .3
Ezp = .3


m_old = np.inf
Ecc_old = np.inf
Ev_old = np.inf
Ezi_old = np.inf
Ezf_old = np.inf
Ezp_old = np.inf
g_old = np.inf


diff = np.inf
tol = .001

diff_vec = []
elbo_vec = []

k = 0

while diff > tol:
    
    #update qx
    Lambda, Lambda_m = update_qc(T, mu_0, covar, Ev, E_gamma, Ezi, Ezf, Ezp)
    m, Ecc = get_c_expects(Lambda, Lambda_m) 

    #update qz
    Ev = update_qv(m)
        
    #update q_omega
    b, g, E_gamma = update_q_gamma(Ecc)
   
    Ezi = update_qzi(mu_0, covar, m, Ezp, Ezf, pi)
 
    Ezf = update_qzf(mu_0, covar, m, Ezi, Ezp, pf)
    
    Ezp = update_qzp(mu_0, covar, m, Ezi, Ezf, pp)

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


    diff = np.amax( [m_diff, Ecc_diff, Ev_diff, 
                     g_diff, Ezi_diff, Ezf_diff, Ezp_diff] )
    diff_vec.append(diff)

    elbo = get_elbo(T,  mu_0, covar, m, Ecc, Ev, E_gamma)
    elbo_vec.append(elbo)

    k+=1


print('m:', m)
Sigma = Ecc-np.outer(m,m)
print('Sigma:')
print(Sigma)
print('Ev:', Ev)
print('g:', g)  
print('Ezi:', Ezi)
print('Ezf:', Ezf)
print('Ezp:', Ezp)

print('elbo:', elbo)
print('sqrt(Ecc)', np.sqrt(Ecc))

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

