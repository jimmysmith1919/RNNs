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


    c[0] = np.random.normal(zf[0]*mu_0+zi[0]*(2*zp[0]-1),
                            np.sqrt(covar[0,0]))
    v[0] = v_gen = np.random.binomial(1,  expit(c[0]))
    for t in range(1,T):
        c[t] = np.random.normal(zf[t]*c[t-1]+zi[t]*(2*zp[t]-1),
                                np.sqrt(covar[t,t]))
        v[t] = np.random.binomial(1,  expit(c[t]))
    
    return c, v, zi, zf, zp

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

def update_q_gamma(Ecc):
    g = np.sqrt(np.diag(Ecc))
    b = np.ones(len(g))
    E_gamma = b/(2*g)*np.tanh(g/2)
    return b, g, E_gamma
##############################################################

def update_qv(Ec): 
    return expit(Ec)
############################################################

def update_zi(mu_0, covar, Ec, Ezp, Ezf, pi):
    diag = np.diag(covar)

    Ec_minus = np.zeros(len(Ec))
    Ec_minus[0] = mu_0
    Ec_minus[1:] = Ec[:-1]
    
    value = 1/diag*Ec*(2*Ezp-1)
    value += -1/diag*Ezf*Ec_minus*(2*Ezp-1)-1/2*1/diag
    value += np.log( pi/(np.ones(len(pi))-pi) )    ###will need to fix
    return expit(value)

def update_zf(mu_0, covar, Ec, Ecc, Ezi, Ezp, pf):
    diag = np.diag(covar)
    
    Ecc_minus_1 = np.zeros(len(Ec))
    Ecc_minus_1[0] = Ec[0]*mu_0
    Ecc_minus_1[1:] = np.diag(Ecc, k=-1)
    
    value = 1/diag*Ecc_minus_1

    Ec_min1_sq = np.zeros(len(Ec))
    Ec_min1_sq[0] = mu_0**2
    Ec_min1_sq[1:] = diag[:-1]

    value += -1/2*1/diag*Ec_min1_sq

    Ec_min1 = np.zeros(len(Ec))
    Ec_min1[0] = mu_0
    Ec_min1[1:] = Ec[:-1]

    value += -1/diag*Ec_min1*Ezi*(2*Ezp-1)
    value += np.log( pf/(np.ones(len(pf))-pf)  )
    return expit(value)

def update_zp(mu_0, covar, Ec, Ezi, Ezf, pp):
    diag = np.diag(covar)
    Ec_minus = np.zeros(len(Ec))
    Ec_minus[0] = mu_0
    Ec_minus[1:] = Ec[:-1]

    value = 2/diag*Ec*Ezi
    value += -2/diag*Ezf*Ec_minus*Ezi
    value += np.log( pp/(np.ones(len(pp))-pp)  )
    return expit(value)
                    

###########################################################
##ELBO calculation###


def elbo_c(mu_0, m, covar, Ecc):
    diag = np.diag(covar)
    Ecc_diag = np.diag(Ecc)

    value = np.log( 1/np.sqrt(2*np.pi*diag) )
    value += -1/2*( 1/diag )*Ecc_diag

    term = 1/diag
    term[0] *= m[0]*mu_0
    term[1:] *= np.diag(Ecc,-1)
    value += term

    term = -1/2*1/diag
    term[0] *= mu_0**2
    term[1:] *= np.diag(Ecc)[:-1]

    value += term

    print('elbo_c:', np.sum(value))
    return np.sum(value)



def elbo_v(Ev, m):
    ones = np.ones(len(Ev))
    value = np.log(1/2)*ones + (Ev-1/2*ones)*m
    print('elbo_v:', np.sum(value))
    return np.sum(value)

def elbo_gamma1(E_gamma, Ecc):
    print('elbo_gamma1:', np.sum( -1/2*E_gamma*np.diag(Ecc) ))
    return np.sum( -1/2*E_gamma*np.diag(Ecc) )


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

def entropy_v(Ev):
    value = -Ev*np.log(Ev)-(np.ones(len(Ev))-Ev)*np.log(np.ones(len(Ev))-Ev)
    print('entropy_v:', np.sum(value))
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

def get_elbo(T,  mu_0, covar, m, Ecc, Ev, E_gamma):
    #NOTE: currently excludes several PG terms
    elbo = elbo_c(mu_0, m, covar, Ecc)
    elbo += elbo_v(Ev, m)
    elbo += elbo_gamma1(E_gamma, Ecc)
    #elbo += elbo_gamma2(E_gamma)
    elbo += elbo_gamma2(Ecc)
    elbo += entropy_c(T, m, Ecc)
    elbo += entropy_v(Ev)
    elbo += entropy_gamma(Ecc)
    return elbo

np.random.seed(0)

T=6

#var = np.random.uniform(.1, .5, size=T)
covar = np.identity(T)*.3*np.ones(T)#var
mu_0 = 1 
pi = np.ones(T)*.5 
pf = np.ones(T)*.5
pp = np.ones(T)*.5

        
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
E_gamma = .3*np.ones(T)
Ev = .8*np.ones(T)
Ezi = .5*np.ones(T)
Ezf = .5*np.ones(T)
Ezp = .5*np.ones(T)

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

    #elbo = get_elbo(T,  mu_0, covar, m, Ecc, Ev, E_gamma)
    #elbo_vec.append(elbo)

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

