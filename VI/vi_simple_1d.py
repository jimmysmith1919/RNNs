import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt


def update_qx(z, mu, var, E_omega):
    s2 = 1/(1/var+E_omega)
    m = ( (1/var)*mu + z-1/2 )*s2
    Exx = m**2 + s2
    return  s2, m, Exx

def update_q_omega(Exx):
    b = 1
    c = np.sqrt(Exx)
    E_omega = b/(2*c)*np.tanh(c/2)
    return b, c, E_omega

def get_elbo(z, mu, var, param_vec, Exx, E_omega):
    #NOTE: excludes expectations of PG distributions (prior and post)
    s2 = param_vec[0]
    Ex = param_vec[1]
    elbo = np.log( 1/np.sqrt((2*np.pi*var)) )-1/(2*var)*Exx
    elbo += mu/var*Ex-(mu**2)/(2*var)
    elbo += np.log(1/2) + (z-1/2)*Ex -1/2*E_omega*Exx
    elbo += np.log(.5*np.pi**2)-.5*np.pi**2*E_omega
    print(np.log(.5*np.pi**2)-.5*np.pi**2*E_omega)
    elbo += 1/2*(1+np.log(2*np.pi*s2))
    elbo += 1-np.log(.5*np.pi**2+Exx/(4*np.pi**2))
    print(1-np.log(.5*np.pi**2+Exx/(4*np.pi**2)))
    return elbo


np.random.seed(0)

mu = 1
var = .3
x_gen = np.random.normal(mu, np.sqrt(var))
z_gen = np.random.binomial(1,  expit(x_gen))


#Initialize q_omega
E_omega = .5
param_vec_old = np.inf*np.ones(4)


error = np.inf
tol = .00001

elbo_vec = []
m_vec = []
s2_vec = []

k = 0


while error > tol:
    param_vec = np.zeros(4) #[qx_s2, qx_m, qw_b, qw_c] 
    
    #update qx
    param_vec[0], param_vec[1], Exx = update_qx(z_gen, mu, var, E_omega)
    s2_vec.append(param_vec[0])
    m_vec.append(param_vec[1])
    
    #update q_omega
    param_vec[2], param_vec[3], E_omega = update_q_omega(Exx)


    error = np.amax( np.absolute(param_vec-param_vec_old) )
    param_vec_old = param_vec
    print('err:', error)
    print(param_vec)

    elbo = get_elbo(z_gen, mu, var, param_vec, Exx, E_omega)
    elbo_vec.append(elbo)
    print('elbo:',elbo)
    k+=1



plt.plot(np.arange(k), elbo_vec)
plt.xlabel('Iteration')
plt.ylabel('ELBO')
plt.title('ELBO convergence (excludes some PG terms)')
plt.savefig('ELBO_0.png')
plt.show()
plt.close()
plt.plot(np.arange(k), m_vec, label='m')
plt.plot(np.arange(k), s2_vec, label = 's^2')
plt.xlabel('Iteration')
plt.legend()
plt.title('Posterior q(x) parameters m and s^2')
plt.savefig('Params.png')
plt.show()
