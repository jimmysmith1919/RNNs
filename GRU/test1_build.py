import numpy as np
from PG_int import pgpdf

def build_prec_x(inv_var,W,pg):
    return np.diag(inv_var)+ (W.T @ np.diag(pg) @ W)

def build_prec_muT(x0, u, inv_var, z, omega, W, U,b,T,d):
    val =  W.T @ (z-1/2)
    val +=  (np.diag(inv_var) @ x0).reshape(T,d,1)
    val +=  W.T @ -(omega*(U @ u + b)) 
    return  val

def log_like(x0,inv_var,x, z, omega, u, W, U, b, bpg, T, d):
    
    x0 = x0.reshape(T,d,1)
    sum = -1/2*(x-x0).transpose(0,2,1) @ np.diag(inv_var) @ (x-x0)
    sum += d*np.log(1/2)
    sum += (W @ x + U @ u + b).transpose(0,2,1) @  (z-1/2) 
    sum += -1/2*x.transpose(0,2,1) @ W.T @ np.diag(omega[0,:,0]) @ W @ x
    sum += -(omega * (U @ u + b) ).transpose(0,2,1) @ W @ x
    sum += np.sum(pgpdf(omega, bpg, 0, 20))
    return sum

