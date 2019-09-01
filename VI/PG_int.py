import numpy as np

from scipy.special import gamma, gammaln, beta
from scipy.linalg.lapack import dpotrs
from scipy.linalg import solve_triangular
from scipy.integrate import simps
from scipy import integrate
from scipy import optimize


def _psi_n(x, n, b):
    """
    Compute the n-th term in the infinite sum of
    the Jacobi density.
    """
    return 2**(b-1) / gamma(b) * (-1)**n * \
    np.exp(gammaln(n+b) -
           gammaln(n+1) +
           np.log(2*n+b) -
           0.5 * np.log(2*np.pi*x**3) -
           (2*n+b)**2 / (8.*x))

def _tilt(omega, b, psi):
    """
    Compute the tilt of the PG density for value omega
    and tilt psi.
    :param omega: point at which to evaluate the density
    :param psi: tilt parameter
    """
    return np.cosh(psi/2.0)**b * np.exp(-psi**2/2.0 * omega)


def pgpdf(omega, b, psi, trunc=200):
    """
    Approximate the density log PG(omega | b, psi) using a
    truncation of the density written as an infinite sum.
    :param omega: point at which to evaluate density
    :param b:   first parameter of PG
    :param psi: tilting of PG
    :param trunc: number of terms in sum
    """
    ns = np.arange(trunc)
    psi_ns = np.array([_psi_n(omega, n, b) for n in ns])
    pdf = np.sum(psi_ns, axis=0)

    # Account for tilting
    pdf *= _tilt(omega, b, psi)

    return pdf

def pgmean(b, psi):
    return b / (2.*psi) * np.tanh(psi/2.)



def qdf_log_pdf(omega, b, psi, c):
    pdf = pgpdf(omega, b, psi)
    if np.isclose(pdf,0, atol=1e-07) == 1:
        return 0
    else:
        return pgpdf(omega,b,c)*np.log(pdf)


def qdf_log_pdf_vec(omega, b, psi, c):
    pdf = pgpdf(omega, b, psi, 10)
    with np.errstate(divide='ignore'):
        logpdf = np.log(pdf)
    logpdf = np.nan_to_num(logpdf,copy=False)
    qdf = pgpdf(omega,b,c,10)
    return qdf*logpdf, qdf


def entropy_q(omega, b, psi):
    pdf = pgpdf(omega, b, psi)
    if np.isclose(pdf, 0, atol=1e-07) == 1:
        return 0
    else:
        return -pdf*np.log(pdf)
'''
def entropy_q_vec(omega, b, psi):
    pdf = pgpdf(omega, b, psi)
    with np.errstate(divide='ignore'):
        logpdf = np.log(pdf)
    logpdf = np.nan_to_num(logpdf,copy=False)
    return -pdf*logpdf
'''
def entropy_q_vec(qdf):
    with np.errstate(divide='ignore'):
        logqdf = np.log(qdf)
    logqdf = np.nan_to_num(logqdf,copy=False)
    return -qdf*logqdf









'''
k= integrate.quad(pgpdf, 0, np.inf, 
                     args=(1, psi), 
                 epsabs=1e-4,   epsrel = 0)[0]

print(k)
'''
'''
h = .01
L = 100
num = int(L/h)

x = np.linspace (.00001,L,num )
y = pgpdf(x, 1, 0, trunc=200)
print(y)

val = np.trapz(y, x)
print(val)
'''

'''
import time

h = .001
L = 10
num = int(L/h)

psi = 0
c=3


T = 4
d=3

a = .00001*np.ones((T,d,d))
b = L * np.ones((T,d,d))
psi = 12
psi_vec = psi*np.ones((T,d,d))

start = time.time()
x = np.linspace (a,b,num )

#y = qdf_log_pdf_vec(x, 1, psi, c)
y = entropy_q_vec(x,1,psi_vec)

val1 = np.trapz(y, x, axis=0)
end = time.time()
print('Trapz:',val1)
print('Time:', end-start)

start = time.time()
val2= integrate.quad(entropy_q, 0, np.inf, 
                     args=(1, psi), 
                 epsabs=1e-4,   epsrel = 0)[0]
end= time.time()
print('quad:',val2)
print('Time:', end-start)
'''

'''
start = time.time()
x = np.linspace (.00001,L,num )
#y = qdf_log_pdf_vec(x, 1, psi, c)
y = entropy_q_vec(x,1,psi)

val1 = np.trapz(y, x)
end = time.time()
print('Trapz:',val1)
print('Time:', end-start)

start = time.time()
val2= integrate.quad(entropy_q, 0, np.inf, 
                     args=(1, psi), 
                 epsabs=1e-4,   epsrel = 0)[0]
end= time.time()
print('quad:',val2)
print('Time:', end-start)
'''
