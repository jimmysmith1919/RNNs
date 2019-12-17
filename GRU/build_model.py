import numpy as np
from PG_int import pgpdf
import sys

def build_prec_x(inv_var, Sigma_y_inv, Wy, Wz, pg_z, z, Wr,
                 pg_r, r, Wp, pg_v, v, T, d):
    prec = np.zeros((T*d,T*d))
    Sigma_inv = np.diag(inv_var)
    for t in range(0,T-1):
        #Diagonal (Block)
        prec[t*d:t*d+d,t*d:t*d+d] = Wy.T @ Sigma_y_inv @ Wy

        
        
        prec[t*d:t*d+d,t*d:t*d+d]+=Sigma_inv+np.outer(1-z[t+1,:,0],
                                                     1-z[t+1,:,0])*Sigma_inv

        prec[t*d:t*d+d,t*d:t*d+d] += Wz.T @ np.diag(pg_z[t+1,:,0]) @ Wz

        prec[t*d:t*d+d,t*d:t*d+d] += Wr.T @ np.diag(pg_r[t+1,:,0]) @ Wr

        rW = Wp*r[t+1,:,0]
        prec[t*d:t*d+d,t*d:t*d+d] += 4*(rW.T @ np.diag(pg_v[t+1,:,0]) @ rW)
        
        #Off-Diagonal
        cross = -np.outer(1-z[t+1,:,0], np.ones(d))*Sigma_inv
        prec[t*d:t*d+d,t*d+d:t*d+2*d] = cross
        prec[t*d+d:t*d+2*d, t*d:t*d+d] = cross
        
    prec[(T-1)*d:(T-1)*d+d,(T-1)*d:(T-1)*d+d] = Sigma_inv + (Wy.T @ Sigma_y_inv
                                                            @ Wy)
    return prec



def build_prec_x_kalman(inv_var, Sigma_y_inv, Wy, Wz, pg_z, z, Wr,
                 pg_r, r, Wp, pg_v, v, T, d):
    
    Sigma_inv = np.diag(inv_var)

    J_ini = Sigma_inv
    
    J_dyn_11 = np.zeros((T-1, d, d))
    J_dyn_22 = np.zeros((T-1, d, d))
    J_dyn_22 += Sigma_inv
    
    
    J_dyn_21 = np.zeros((T-1, d, d))

    J_obs = np.zeros((T,d,d))
    J_obs += Wy.T @ Sigma_y_inv @ Wy


    for t in range(0,T-1):
        #Diagonal (Block)
        J_dyn_11[t]+=np.outer(1-z[t+1,:,0], 1-z[t+1,:,0])*Sigma_inv

        J_dyn_11[t] += Wz.T @ np.diag(pg_z[t+1,:,0]) @ Wz

        J_dyn_11[t] += Wr.T @ np.diag(pg_r[t+1,:,0]) @ Wr

        rW = Wp*r[t+1,:,0]
        J_dyn_11[t] += 4*(rW.T @ np.diag(pg_v[t+1,:,0]) @ rW)
            
        #Off-Diagonal
        cross = -np.outer(1-z[t+1,:,0], np.ones(d))*Sigma_inv
        J_dyn_21[t] = cross
    
    return J_dyn_11, J_dyn_22, J_dyn_21, J_obs, J_ini



def build_prec_muT(x0, u, inv_var, y, Sigma_y_inv, Wy, by,
                   z, pg_z, Wz, Uz, bz, r, pg_r, Wr, Ur, br,
                   v, Wp, Up, bp, pg_v, T, d):
    val = np.zeros((T*d,1))

    val[:,0] += (Wy.T @ Sigma_y_inv @ (y-by)).ravel()
        
    val[:d,0] +=  inv_var*((1-z[0,:,0])*x0)

    z2v =  z*(2*v-1)
    val[:,0] += (z2v[:,:,0]*inv_var).ravel()
    
    val[:(T-1)*d,0] +=  (Wz.T @ (z[1:,:,:]-1/2)).ravel()
    val[:(T-1)*d,0] +=  (Wz.T @ -(pg_z[1:,:,:]*(Uz @ u[1:,:,:] + bz)) ).ravel()

    val[:(T-1)*d,0] +=  (Wr.T @ (r[1:,:,:]-1/2)).ravel()
    val[:(T-1)*d,0] +=  (Wr.T @ -(pg_r[1:,:,:]*(Ur @ u[1:,:,:] + br)) ).ravel()

    
    rW = r[1:,:,:].transpose(0,2,1)*Wp
    rWT = rW.transpose(0,2,1)
    val[:(T-1)*d,0] +=  2*(rWT @ (v[1:,:,:]-1/2)).ravel()
    val[:(T-1)*d,0] +=  4*(rWT @ -(pg_v[1:,:,:]*(Up @ u[1:,:,:] +
                                                  bp)) ).ravel()
    return  val

def build_prec_muT_kalman(x0, u, inv_var, y, Sigma_y_inv, Wy, by,
                   z, pg_z, Wz, Uz, bz, r, pg_r, Wr, Ur, br,
                   v, Wp, Up, bp, pg_v, T, d):

    h_obs = Wy.T @ Sigma_y_inv @ (y-by)
    h_obs = h_obs.reshape(T,d)
    

    z2v =  z*(2*v-1)
    h_ini = inv_var*( ((1-z[0,:,0])*x0)+z2v[0,:,0] )


    
    h_dyn_2 = np.zeros((T-1,d))
   
    h_dyn_2 = z2v[1:,:,0]*inv_var


    h_dyn_1 = np.zeros((T-1,d))
    
    h_dyn_1 += (Wz.T @ (z[1:,:,:]-1/2))[:,:,0]
    h_dyn_1 += (Wz.T @ -(pg_z[1:,:,:]*(Uz @ u[1:,:,:] + bz)))[:,:,0]

    h_dyn_1 += (Wr.T @ (r[1:,:,:]-1/2))[:,:,0]
    h_dyn_1 += (Wr.T @ -(pg_r[1:,:,:]*(Ur @ u[1:,:,:] + br)) )[:,:,0]

    rW = r[1:,:,:].transpose(0,2,1)*Wp
    rWT = rW.transpose(0,2,1)

    h_dyn_1 += (2*(rWT @ (v[1:,:,:]-1/2)))[:,:,0]
    h_dyn_1 += (4*(rWT @ -(pg_v[1:,:,:]*(Up @ u[1:,:,:] + bp)) ) )[:,:,0] 
    
    return  h_dyn_1, h_dyn_2, h_obs, h_ini



def build_z_param(h, v, inv_var, W, U, u, b,d):
    h_minus_sq = (h[:-1,:,:])**2
    h_h_minus = h[:-1,:,:]*h[1:,:,:]
    
    fz = W @ h[:-1,:,:] + U @ u + b
    fz += (-h_h_minus + h[1:,:,:]*(2*v-1) + 1/2*h_minus_sq - 1/2)*inv_var
    return fz

def build_rd_param(h,u,v,gamma,r,Wp,Up,bp,Wr,Ur,br,dim):
    hmin_Wdd = h[:-1,dim,:]*Wp[dim, dim]
    param = 2*(v[:,dim,:]-1/2)*hmin_Wdd

    Wp_min_d = np.delete(Wp,dim,1)
    r_min_d = np.delete(r,dim,1)
    h_min_d = np.delete(h,dim,1)
    rh_min = r_min_d*h_min_d[:-1,:,:]
    fp_min = Wp_min_d[dim,:] @ rh_min+Up[dim,:]@ u + bp[dim,:]

    param += -2*gamma[:,dim,:]*( hmin_Wdd**2+2*hmin_Wdd*fp_min ) 
    param += Wr[dim,:] @ h[:-1,:,:] + Ur[dim,:] @ u + br[dim,:] 
    return param

def build_v_param(h,z,rh,u,Wp,Up, bp, inv_var,d):
    fv = 2*(Wp @ rh[:-1,:,:] + Up @ u + bp)
    
    hz = h[1:,:,:]*z
    fv += 2*hz*inv_var.reshape(d,1)
    return fv

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


