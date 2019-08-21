import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt


def update_qc(T, t, mu, covar, Ev, E_gamma):
    Lambda = np.zeros((T,T))
    #Digaonal
    Lambda[t-1,t-1] = 1/covar[t-1,t-1]+1/covar[t,t]
    Lambda[t,t] =  1/covar[t,t]+1/covar[t+1,t+1]+E_gamma
    Lambda[t+1,t+1] = 1/covar[t+1,t+1]
    #Crossterms
    Lambda[t-1,t] = -1/covar[t,t]
    Lambda[t, t-1] = Lambda[t-1,t]
    
    Lambda[t,t+1] = -1/covar[t+1,t+1]
    Lambda[t+1,t] = Lambda[t,t+1]

    #lambda_m terms
    Lambda_m = np.zeros(T)
    Lambda_m[t-1] = 1/covar[t-1,t-1]*mu #will need to update mu term
    Lambda_m[t] = Ev-1/2
    return  Lambda, Lambda_m

def get_c_expects(Lambda, Lambda_m):
    #will need to update with message passing
    Sigma = np.linalg.inv(Lambda)
    m = Sigma @ Lambda_m
    Ecc = np.outer(m, m) + Sigma
    return m, Ecc

def update_q_gamma(Ecc):
    b = 1
    c = np.sqrt(Ecc)
    E_gamma = b/(2*c)*np.tanh(c/2)
    return b, c, E_gamma

def update_qv(Ec):
    return expit(Ec)

def get_elbo(T, t, mu, covar,  m, Ecc, Ev, E_gamma):
    #NOTE: excludes expectations of PG distributions (prior and post)
    
    Sigma = Ecc-np.outer(m, m) #need to fix
    
    elbo = 0
    for j in range(t-1,t+2):
        elbo += np.log( 1/np.sqrt((2*np.pi*covar[j,j])))
        elbo += -1/(2*covar[j,j])*Ecc[j,j]
    
    #need to fix zero term later
    ####
    elbo += mu/covar[t-1,t-1]*m[t-1]-(mu**2)/(2*covar[t-1,t-1])
    for j in range(t,t+2):
        elbo += 1/covar[j,j]*Ecc[j,j-1]
        elbo += -1/(2*covar[j,j])*Ecc[j,j]
    ###                               
    
    elbo += np.log(1/2) + (Ev-1/2)*m[t] -1/2*E_gamma*Ecc[t,t]
    #using determinant, need to fix
    elbo += T/2*(1+np.log(2*np.pi))+1/2*np.log(np.linalg.det(Sigma))
    elbo += -Ev*np.log(Ev)-(1-Ev)*np.log(1-Ev)
    return elbo


np.random.seed(0)


covar = np.diag([.1, .2, .3])
mu_0 = 1
c0 = np.random.normal(mu_0, np.sqrt(covar[0,0]))
c1 = np.random.normal(c0, np.sqrt(covar[1,1]))
v_gen = np.random.binomial(1,  expit(c1))
c2 = np.random.normal(c1, np.sqrt(covar[2,2]))



print('Mu=',mu_0)
print('c0:', c0)
print('c1:', c1)
print('c2:', c2)
print('covar=',covar)
print('Sigmoid(c1)=', expit(c1))

#Initialize q_omega
E_gamma = .3
Ev = .8

T = 3

param_vec_old = np.ones(3)*np.inf
m_old = np.ones(T)*np.inf
Ecc_old = np.ones((T,T))*np.inf

diff = np.inf
tol = .00001

diff_vec = []
elbo_vec = []
m_vec = []
s2_vec = []
p_vec = []
c_vec = []

k = 0


t = 1

while diff > tol:
    param_vec = np.zeros(3) #[qv_p, qgam_b, qgam_c] 
    
    #update qc
    Lambda, Lambda_m = update_qc(T, t, mu_0, covar, Ev, E_gamma)
    m, Ecc = get_c_expects(Lambda, Lambda_m) 

    #update qv
    Ev = update_qv(m[t])
    param_vec[0] = Ev
    p_vec.append(Ev)
        
    #update q_gamma
    b, c, E_gamma = update_q_gamma(Ecc[t,t])
    param_vec[1] = b
    param_vec[2] = c
    c_vec.append(c)
    


    
    m_diff = np.amax(np.absolute(m-m_old))
    Ecc_diff = np.amax(np.absolute(Ecc-Ecc_old))
    m_old = m
    Ecc_old = Ecc

    diff = np.amax( np.absolute(param_vec-param_vec_old) )
    param_vec_old = param_vec
    diff = np.amax([diff, m_diff, Ecc_diff])
    diff_vec.append(diff)

    
    elbo = get_elbo(3, 1, mu_0, covar,  m, Ecc, Ev, E_gamma)
    elbo_vec.append(elbo)

    k+=1



print('m0:', m[0])
print('m1:', m[1])
print('m2:', m[2])
Sigma = Ecc-np.outer(m,m)
print('Sigma[0,0]:', Sigma[0,0])
print('Sigma[1,1]:', Sigma[1,1])
print('Sigma[2,2]:', Sigma[2,2])
print('p:', param_vec[0])
print('c:', param_vec[2])  




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

'''
plt.plot(np.arange(k), m_vec, label='m')
plt.plot(np.arange(k), s2_vec, label = 's^2')
plt.plot(np.arange(k), p_vec, label = 'p') 
plt.plot(np.arange(k), c_vec, label = 'c')
plt.xlabel('Iteration')
plt.legend()
plt.title('Parameters: q(c),  m and s^2; q(v), p, q(\u03B3)')
plt.savefig('Params.png')
plt.show()
'''
