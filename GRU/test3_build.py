import numpy as np
from PG_int import pgpdf
import sys

def build_prec_x(inv_var,W,pg,z,T,d):
    prec = np.zeros((T*d,T*d))
    for t in range(0,T-1):
        Sigma_inv = np.diag(inv_var)
        prec[t*d:t*d+d,t*d:t*d+d]=Sigma_inv+np.outer(1-z[t+1,:,0],
                                                     1-z[t+1,:,0])*Sigma_inv+(
                                               W.T @ np.diag(pg[t+1,:,0]) @ W)
        

        cross = -np.outer(1-z[t+1,:,0], np.ones(d))*Sigma_inv
        prec[t*d:t*d+d,t*d+d:t*d+2*d] = cross
        prec[t*d+d:t*d+2*d, t*d:t*d+d] = cross
        
    prec[(T-1)*d:(T-1)*d+d,(T-1)*d:(T-1)*d+d] = Sigma_inv
    return prec

def build_prec_muT(x0, u, inv_var, z, omega, W, U,b,T,d):
    val = np.zeros((T*d,1))
    val[:(T-1)*d,0] +=  (W.T @ (z[1:,:,:]-1/2)).ravel()
    val[:d,0] +=  inv_var*((1-z[0,:,0])*x0)
    val[:(T-1)*d,0] +=  (W.T @ -(omega[1:,:,:]*(U @ u[1:,:,:] + b)) ).ravel()
    return  val

def build_z_param(h, inv_var, W, U, u, b,d):
    h_minus_sq = (h[:-1,:,:])**2
    h_h_minus = h[:-1,:,:]*h[1:,:,:]

    fz = W @ h[:-1,:,:] + U @ u + b 
    fz += (1/2*h_minus_sq-h_h_minus)*inv_var
    return fz


def log_like(x0,inv_var,x, z, omega, u, W, U, b, bpg, T, d):
    
    x0 = x0.reshape(1,d,1)
    sum = -1/2*(x-x0).transpose(0,2,1) @ np.diag(inv_var) @ (x-x0)
    sum += -d/2*np.log(2*np.pi)-1/2*np.log(np.prod(1/inv_var))
    sum += d*np.log(1/2)
    sum += (W @ x + U @ u + b).transpose(0,2,1) @  (z-1/2) 
    sum += -1/2*x.transpose(0,2,1) @ W.T @ np.diag(omega[0,:,0]) @ W @ x
    sum += -(omega * (U @ u + b) ).transpose(0,2,1) @ W @ x
    sum += np.sum(pgpdf(omega, bpg, 0, 20))
    return sum


    
