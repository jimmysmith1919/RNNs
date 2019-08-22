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

def entropy_q(omega, b, psi):
    pdf = pgpdf(omega, b, psi)
    if np.isclose(pdf, 0, atol=1e-07) == 1:
        return 0
    else:
        return -pdf*np.log(pdf)



'''
import time
start = time.time()

c = np.ones(5)*.3



print(integrate.quad(qdf_log_pdf, np.zeros(5), np.ones(5)*np.inf, 
                     args=(np.ones(5),np.zeros(5), c), 
                 epsabs=1e-1,   epsrel = 0))
end = time.time()
print(end-start)
'''
'''
import time
start = time.time()


print(integrate.quad(qdf_log_pdf, 0, np.inf, args=(1,0, c),
      epsabs=1e-1))
end = time.time()
print(end-start)
'''




#print(integrate.quad(entropy_q, 0, np.inf, args=(1,c), epsabs=0, epsrel=1e-3))









