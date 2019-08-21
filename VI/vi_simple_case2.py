import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt


def update_qx(mu, var, Ez, E_omega):
    s2 = 1/(1/var+E_omega)
    m = ( (1/var)*mu + Ez-1/2 )*s2
    Exx = m**2 + s2
    return  s2, m, Exx

def update_q_omega(Exx):
    b = 1
    c = np.sqrt(Exx)
    E_omega = b/(2*c)*np.tanh(c/2)
    return b, c, E_omega

def update_qz(Ex):
    return expit(Ex)

def get_elbo(mu, var, param_vec, Exx, E_omega):
    #NOTE: excludes expectations of PG distributions (prior and post)
    s2 = param_vec[0]
    Ex = param_vec[1]
    Ez = param_vec[2]
    elbo = np.log( 1/np.sqrt((2*np.pi*var)) )-1/(2*var)*Exx
    elbo += mu/var*Ex-(mu**2)/(2*var)
    term1 = elbo
    print('E log p(x):', term1)
    elbo += np.log(1/2) + (Ez-1/2)*Ex
    term2 = elbo - term1
    print('E lgo(z)', term2)
    elbo += -1/2*E_omega*Exx + np.log(.5*np.pi**2)-.5*np.pi**2*E_omega
    #elbo += -1/2*E_omega*Exx + np.log(2*np.pi**2)-2*np.pi**2*E_omega
    term3 = elbo-term2-term1
    print('E log(omega)', term3)
    elbo += 1/2*(1+np.log(2*np.pi*s2))
    print('x_ent;', 1/2*(1+np.log(2*np.pi*s2)))
    elbo += -Ez*np.log(Ez)-(1-Ez)*np.log(1-Ez)
    print('z_ent:', -Ez*np.log(Ez)-(1-Ez)*np.log(1-Ez))
    elbo += 1-np.log(.5*np.pi**2+Exx/(4*np.pi**2))    
    #elbo += 1-np.log(2*np.pi**2+Exx/(4*np.pi**2))    
    print('om-ent:', 1-np.log(.5*np.pi**2+Exx/(4*np.pi**2)) )
    print(' ')
    return elbo

np.random.seed(0)

mu = 1
var = .1
x_gen = np.random.normal(mu, np.sqrt(var))
z_gen = np.random.binomial(1,  expit(x_gen))

print('Mu=',mu)
print('Var=',var)
print('Sigmoid(x)=', expit(x_gen))

#Initialize q_omega
E_omega = .5
Ez = .5
param_vec_old = np.inf*np.ones(5)


error = np.inf
tol = .00001

error_vec = []
elbo_vec = []
m_vec = []
s2_vec = []
p_vec = []
c_vec = []

k = 0


while error > tol:
    param_vec = np.zeros(5) #[qx_s2, qx_m, qz_p, qw_b, qw_c] 
    
    #update qx
    param_vec[0], param_vec[1], Exx = update_qx(mu, var, Ez, E_omega)
    s2_vec.append(param_vec[0])
    m_vec.append(param_vec[1])
    Ex = param_vec[1]

    #update qz
    param_vec[2] = update_qz(Ex)
    Ez = param_vec[2]
    p_vec.append(Ez)
    
    #update q_omega
    param_vec[3], param_vec[4], E_omega = update_q_omega(Exx)
    c_vec.append(param_vec[4])

    error = np.amax( np.absolute(param_vec-param_vec_old) )
    error_vec.append(error)

    param_vec_old = param_vec

    elbo = get_elbo(mu, var, param_vec, Exx, E_omega)
    elbo_vec.append(elbo)

    k+=1



print('m:', param_vec[1])
print('s2:', param_vec[0])
print('p:', param_vec[2])
print('c:', param_vec[4])  




plt.plot(np.arange(k), elbo_vec)
plt.xlabel('Iteration')
plt.ylabel('ELBO')
plt.title('ELBO convergence (excludes some PG terms)')
plt.savefig('ELBO_0.png')
plt.show()
plt.close()

plt.plot(np.arange(k), error_vec)
plt.xlabel('Iteration')
plt.ylabel('Max Parameter Difference')
plt.savefig('Error.png')
plt.show()
plt.close()

plt.plot(np.arange(k), m_vec, label='m')
plt.plot(np.arange(k), s2_vec, label = 's^2')
plt.plot(np.arange(k), p_vec, label = 'p') 
plt.plot(np.arange(k), c_vec, label = 'c')
plt.xlabel('Iteration')
plt.legend()
plt.title('Parameters: q(x),  m and s^2; q(z), p, q(u03c9')
plt.savefig('Params.png')
plt.show()
