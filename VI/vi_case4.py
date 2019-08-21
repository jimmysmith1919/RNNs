import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt


def generate(T, mu_0, covar):
    c = np.zeros(T)
    v = np.zeros(T)
    
    c[0] = np.random.normal(mu_0, np.sqrt(covar[0,0]))
    v[0] = v_gen = np.random.binomial(1,  expit(c[0]))
    for t in range(1,T):
        c[t] = np.random.normal(c[t-1], np.sqrt(covar[t,t]))
        v[t] = np.random.binomial(1,  expit(c[t]))
    return c, v

##update c ##

def update_diag(t, covar, E_gamma):
    return  1/covar[t,t]+1/covar[t+1,t+1]+E_gamma[t]

def update_crossterms(t, covar):
    return -1/covar[t,t]

def update_Lambda_M(t, Ev):
    return Ev[t]-1/2
    
def update_L_m0(mu_0, covar, Ev):
    #takes initial condition mu_0 into account
    return (Ev[0]-1/2) + 1/covar[0,0]*mu_0

def update_LambdaTT(t, covar, E_gamma):
    #non next state covariance
    return  1/covar[t,t]+E_gamma[t]

def update_qc(T, mu_0, covar, Ev, E_gamma):
    Lambda = np.zeros((T,T))
    Lambda_m = np.zeros(T)

    #c_0
    Lambda[0,0] = update_diag(0, covar, E_gamma)
    Lambda_m[0] = update_L_m0(mu_0, covar, Ev)
    
    #c_1:c_T-1
    for t in range(1,T-1):    
        #Digaonal
        Lambda[t,t] =  update_diag(t, covar, E_gamma)
    
        #Crossterms
        Lambda[t-1,t] = update_crossterms(t, covar)
        Lambda[t, t-1] = Lambda[t-1,t]
    
        #lambda_m terms
        Lambda_m[t] = update_Lambda_M(t, Ev)
    
    #c_T
    t = T-1
    Lambda[t,t] = update_LambdaTT(t, covar, E_gamma)
    Lambda[t-1,t] = update_crossterms(t, covar)
    Lambda[t, t-1] = Lambda[t-1,t]
    Lambda_m[t] = update_Lambda_M(t, Ev)

    return  Lambda, Lambda_m

def get_c_expects(Lambda, Lambda_m):
    #will need to update with message passing
    Sigma = np.linalg.inv(Lambda)
    m = Sigma @ Lambda_m
    Ecc = np.outer(m, m) + Sigma
    return m, Ecc

#######

def update_q_gamma(Ecc):
    g = np.sqrt(np.diag(Ecc))
    b = np.ones(len(g))
    E_gamma = b/(2*g)*np.tanh(g/2)
    return b, g, E_gamma


def update_qv(Ec): 
    return expit(Ec)


##ELBO calculation###
def elbo_c(mu_0, m, covar, Ecc):
    value = np.sum( np.log( 1/np.sqrt(2*np.pi*np.diag(covar)) ) )
    value += np.sum( -1/2*( 1/np.diag(covar) )*np.diag(Ecc) )
    
    short_diag = 1/np.diag(covar)
    short_diag = short_diag[1:]
    value += np.sum(short_diag*np.diag(Ecc, -1))
    value += 1/covar[0,0]*mu_0*m[0]
    
    Ecc_minus = np.diag( Ecc )[:-1]
    value += np.sum( -1/2*short_diag*Ecc_minus )
    value += -1/2*1/covar[0,0]*mu_0**2
    print('elbo_c:', value)
    return value

def elbo_v(Ev, m):
    ones = np.ones(len(Ev))
    value = np.log(1/2)*ones + (Ev-1/2*ones)*m
    print('elbo_v:', np.sum(value))
    return np.sum(value)

def elbo_gamma1(E_gamma, Ecc):
    print('elbo_gamma1:', np.sum( -1/2*E_gamma*np.diag(Ecc) ))
    return np.sum( -1/2*E_gamma*np.diag(Ecc) )

def elbo_gamma2(E_gamma):
    value = np.log(.5*np.pi**2)*np.ones(len(E_gamma))-.5*np.pi**2*E_gamma
    #value = np.log(2*np.pi**2)*np.ones(len(E_gamma))-2*np.pi**2*E_gamma
    print('elbo_gamma', np.sum(value))
    return np.sum(value)

def entropy_c(T, m, Ecc):
    #need to make more efficient
    Sigma = Ecc-np.outer(m,m)
    print('Entropy_c:', T/2*(1+np.log(2*np.pi))+1/2*np.log( np.linalg.det(Sigma) ))
    return T/2*(1+np.log(2*np.pi))+1/2*np.log( np.linalg.det(Sigma) )

def entropy_v(Ev):
    value = -Ev*np.log(Ev)-(np.ones(len(Ev))-Ev)*np.log(np.ones(len(Ev))-Ev)
    print('entropy_v:', np.sum(value))
    return np.sum(value)
    
def entropy_gamma(Ecc):
    diag = np.diag(Ecc)
    ones = np.ones(len(diag))
    value = ones-np.log( (np.pi**2*ones)/2+diag/(2))
    #value = ones-np.log( 2*np.pi**2*ones+diag/(4*np.pi**2))
    print('gam_entrpy:', np.sum(value))
    print(' ')
    return np.sum(value)


def get_elbo(T,  mu_0, covar, m, Ecc, Ev, E_gamma):
    #NOTE: currently excludes several PG terms
    elbo = elbo_c(mu_0, m, covar, Ecc)
    elbo += elbo_v(Ev, m)
    elbo += elbo_gamma1(E_gamma, Ecc)
    elbo += elbo_gamma2(E_gamma)
    elbo += entropy_c(T, m, Ecc)
    elbo += entropy_v(Ev)
    elbo += entropy_gamma(Ecc)
    return elbo

np.random.seed(0)

T=5

#var = np.random.uniform(.1, .5, size=T)
covar = np.identity(T)*.3*np.ones(T)#var
mu_0 = .3 

        
c,v = generate(T, mu_0, covar)

print('Mu_0=',mu_0)
print('covar=')
print(covar)
print('c:', c)
print('Sigmoid(c):', expit(c))
print('v:', v)
print(' ')


#Initialize q_omega
E_gamma = .3*np.ones(T)
Ev = .8*np.ones(T)


m_old = np.ones(T)*np.inf
Ecc_old = np.ones((T,T))*np.inf
Ev_old = np.ones(T)*np.inf
g_old = np.ones(T)*np.inf


diff = np.inf
tol = .00001

diff_vec = []
elbo_vec = []

k = 0

while diff > tol:
    
    #update qx
    Lambda, Lambda_m = update_qc(T, mu_0, covar, Ev, E_gamma)
    m, Ecc = get_c_expects(Lambda, Lambda_m) 

    #update qz
    Ev = update_qv(m)
        
    #update q_omega
    b, g, E_gamma = update_q_gamma(Ecc)
    
    #convergence check
    m_diff = np.amax(np.absolute(m-m_old))
    m_old = m
    
    Ecc_diff = np.amax(np.absolute(Ecc-Ecc_old))
    Ecc_old = Ecc

    Ev_diff = np.amax(np.absolute(Ev-Ev_old))
    Ev_old = Ev

    g_diff = np.amax(np.absolute(g-g_old))
    g_old = g
    
    diff = np.amax( [m_diff, Ecc_diff, Ev_diff, g_diff] )
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

