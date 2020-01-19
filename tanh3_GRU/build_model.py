
import numpy as np
from PG_int import pgpdf
import sys
from scipy.special import expit
from scipy.stats import norm

def build_prec_x(inv_var, Sigma_y_inv, Wy, Wz, pg_z, z, Wr,
                 pg_r, r, Wp, pg_v, v, T, d, tau, alpha):
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
        vcat = np.argmax(v[t+1], axis=1)
        indgreat3 = vcat ==2
        ind3z = indgreat3*z[t+1,:,0]
        
        prec[t*d:t*d+d,t*d:t*d+d] += (1/alpha**2)*(rW.T @
                               (np.diag(ind3z)*Sigma_inv) @ rW)
        
        
        prec[t*d:t*d+d,t*d:t*d+d] += (1/tau**2)*(rW.T @ np.diag(pg_v[t+1,:,0]+
                                                       pg_v[t+1,:,1]) @ rW)
        
        #Off-Diagonal
        cross = -np.outer(1-z[t+1,:,0], np.ones(d))*Sigma_inv
        cross += -1/alpha*(ind3z*rW).T @ Sigma_inv

        prec[t*d:t*d+d,t*d+d:t*d+2*d] = cross
        prec[t*d+d:t*d+2*d, t*d:t*d+d] = cross
        
    prec[(T-1)*d:(T-1)*d+d,(T-1)*d:(T-1)*d+d] = Sigma_inv + (Wy.T @ Sigma_y_inv
                                                            @ Wy)
    return prec



def build_prec_x_kalman(inv_var, Sigma_y_inv, Wy, Wz, pg_z, z, Wr,
                        pg_r, r, Wp, pg_v, v, T, d, tau, alpha):
    
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
        vcat = np.argmax(v[t+1], axis=1)
        indgreat3 = vcat ==2
        ind3z = indgreat3*z[t+1,:,0]
        
        J_dyn_11[t] += (1/tau**2)*(rW.T @ np.diag(pg_v[t+1,:,0]+
                                                  pg_v[t+1,:,1]) @ rW)
        J_dyn_11[t] += (1/alpha**2)*(rW.T @
                               (np.diag(ind3z)*Sigma_inv) @ rW)
            
        #Off-Diagonal
        cross = -np.outer(1-z[t+1,:,0], np.ones(d))*Sigma_inv
        cross += -1/alpha*(ind3z*rW).T @ Sigma_inv
        J_dyn_21[t] = cross
        
    
    return J_dyn_11, J_dyn_22, J_dyn_21, J_obs, J_ini



def build_prec_muT(x0, u, inv_var, y, Sigma_y_inv, Wy, by,
                   z, pg_z, Wz, Uz, bz, r, pg_r, Wr, Ur, br,
                   v, Wp, Up, bp, pg_v, T, d, tau, alpha):
    
    vcat = np.argmax(v, axis=2)
    
    ind1 = vcat == 0
    ind1 = ind1.reshape(T,d,1)
    ind2 = vcat == 1
    ind2 = ind2.reshape(T,d,1)
    ind3 = vcat == 2
    ind3 = ind3.reshape(T,d,1)

    
    V1 = ind1*-1
    V2= ind2*1
    
    zV1 = z*V1
    zV2 = z*V2
    zI3 = z*ind3
    
    val = np.zeros((T*d,1))
    val[:,0] += (Wy.T @ Sigma_y_inv @ (y-by)).ravel()    

    ###initial cross terms
    val[:d,0] +=  inv_var*((1-z[0,:,0])*x0)
    rW0 = Wp*r[0,:,0]
    val[:d,0] += 1/alpha * x0.T @ (zI3[0,:,0]*rW0).T @ np.diag(inv_var)
    ###
    
    zV12 = zV1+zV2
    val[:,0] += (zV12[:,:,0]*inv_var).ravel()
    
    val[:(T-1)*d,0] +=  (Wz.T @ (z[1:,:,:]-1/2)).ravel()
    val[:(T-1)*d,0] +=  (Wz.T @ -(pg_z[1:,:,:]*(Uz @ u[1:,:,:] + bz)) ).ravel()

    val[:(T-1)*d,0] +=  (Wr.T @ (r[1:,:,:]-1/2)).ravel()
    val[:(T-1)*d,0] +=  (Wr.T @ -(pg_r[1:,:,:]*(Ur @ u[1:,:,:] + br)) ).ravel()
    
    rW = r[1:,:,:].transpose(0,2,1)*Wp
    rWT = rW.transpose(0,2,1)


    
    indgreat2 = ind2+ind3
    Uub = Up @ u[1:,:,:]+bp
    
    val[:(T-1)*d,0] +=  -1/tau*(rWT @ (ind1[1:,:,:]-1/2)).ravel()
    val[:(T-1)*d,0] +=  1/tau*(rWT @ (ind2[1:,:,:]-indgreat2[1,:,:]/2)).ravel()

    val[:(T-1)*d,0] +=  -(1/tau**2)*(rWT @ (
                                  pg_v[1:,:,0].reshape(T-1,d,1)*(Uub+alpha)+
                                  pg_v[1:,:,1].reshape(T-1,d,1)*(Uub-alpha)
                                          )  ).ravel()

    
    IIT3 = (ind3[...,None]*ind3[:,None,:]).reshape(T,d,d)
    zzT = (z[...,None]*z[:,None,:]).reshape(T,d,d)


    
    val[:(T-1)*d,0] += -1/alpha*( ( IIT3[1:]*zzT[1:]*(rWT @ np.diag(inv_var) )
                                 ) @ Uub ).ravel()
    
    
    return  val

def build_prec_muT_kalman(x0, u, inv_var, y, Sigma_y_inv, Wy, by,
                   z, pg_z, Wz, Uz, bz, r, pg_r, Wr, Ur, br,
                          v, Wp, Up, bp, pg_v, T, d, alpha, tau):

    h_obs = Wy.T @ Sigma_y_inv @ (y-by)
    h_obs = h_obs.reshape(T,d)
    
    vcat = np.argmax(v, axis=2)
    ind1 = vcat == 0
    ind1 = ind1.reshape(T,d,1)
    ind2 = vcat == 1
    ind2 = ind2.reshape(T,d,1)
    ind3 = vcat == 2
    ind3 = ind3.reshape(T,d,1)

    V1 = ind1*-1
    V2= ind2*1

    zV1 = z*V1
    zV2 = z*V2
    zI3 = z*ind3

    zV12 = zV1+zV2

    rW = r[1:,:,:].transpose(0,2,1)*Wp
    rWT = rW.transpose(0,2,1)

    indgreat2 = ind2+ind3
    Uub = Up @ u[1:,:,:]+bp

    rW0 = Wp*r[0,:,0]
    h_ini = inv_var*( ((1-z[0,:,0])*x0)+zV12[0,:,0] )
    h_ini += 1/alpha * x0.T @ (zI3[0,:,0]*rW0).T @ np.diag(inv_var)
    
    h_dyn_2 = np.zeros((T-1,d))   
    h_dyn_2 = zV12[1:,:,0]*inv_var


    
    ####I think this is h_dyn1, need to check

    IIT3 = (ind3[...,None]*ind3[:,None,:]).reshape(T,d,d)
    zzT = (z[...,None]*z[:,None,:]).reshape(T,d,d)

    h_dyn_2 += (-1/alpha*( ( IIT3[1:]*zzT[1:]*(rWT @ np.diag(inv_var) )
                                 ) @ Uub ) )[:,:,0]
    #####
    
    h_dyn_1 = np.zeros((T-1,d))
    
    h_dyn_1 += (Wz.T @ (z[1:,:,:]-1/2))[:,:,0]
    h_dyn_1 += (Wz.T @ -(pg_z[1:,:,:]*(Uz @ u[1:,:,:] + bz)))[:,:,0]

    h_dyn_1 += (Wr.T @ (r[1:,:,:]-1/2))[:,:,0]
    h_dyn_1 += (Wr.T @ -(pg_r[1:,:,:]*(Ur @ u[1:,:,:] + br)) )[:,:,0]
 
    h_dyn_1 += (-1/tau*(rWT @ (ind1[1:,:,:]-1/2)))[:,:,0]
    h_dyn_1 += (1/tau*(rWT @ (ind2[1:,:,:]-indgreat2[1,:,:]/2)))[:,:,0]
    h_dyn_1 += (-(1/tau**2)*(rWT @ (
                                  pg_v[1:,:,0].reshape(T-1,d,1)*(Uub+alpha)+
                                  pg_v[1:,:,1].reshape(T-1,d,1)*(Uub-alpha)
                                          )  ) )[:,:,0]

    return  h_dyn_1, h_dyn_2, h_obs, h_ini



def build_z_param(h, v, r, inv_var, u, W, U, b, Wp, Up, bp, d, alpha):
    h_minus_sq = (h[:-1,:,:])**2
    h_h_minus = h[:-1,:,:]*h[1:,:,:]

    
    fp = Wp @ h[:-1,:,:] + Up @ u + bp

    V1 = v[:,:,0]*-1    
    V2 = v[:,:,1]
    V3 = v[:,:,2]*1/alpha*fp[:,:,0]

    V = V1+V2+V3
    V = V.reshape(-1,d,1)
    
    fz = W @ h[:-1,:,:] + U @ u + b
    fz += (-h_h_minus + h[1:,:,:]*V + 1/2*h_minus_sq - 1/2*(V**2))*inv_var
    
    return fz

'''
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
'''

def build_rd_param(h,u,v,gamma,r,Wp,Up,bp,Wr,Ur,br,dim):
    
    hmin_Wd = Wp[:, dim]*h[:-1,dim,:]
    hmin_Wd = np.expand_dims(hmin_Wd, axis=2)
        
    param = np.sum(2*(v-1/2)*hmin_Wd, axis=1)
        
    Wp_min_d = np.delete(Wp,dim,1)
    
    r_min_d = np.delete(r,dim,1)
    
    h_min_d = np.delete(h,dim,1)
    
    rh_min = r_min_d*h_min_d[:-1,:,:]
    
    fp_min = Wp_min_d @ rh_min + Up @ u + bp
    
    param += np.sum( -2*gamma*( hmin_Wd**2+2*hmin_Wd*fp_min ), axis=1 )
     
    param += Wr[dim,:] @ h[:-1,:,:] + Ur[dim,:] @ u + br[dim,:]
    return param

def build_v_param(h,z,rh,u,Wp,Up, bp, inv_var,T,d, alpha, tau):
    

    fp = (Wp @ rh[:-1,:,:] + Up @ u + bp)

    zeta1 = (-fp-alpha)/tau
    zeta2 = (fp-alpha)/tau

    print(zeta1.shape)
    print(zeta1)

    pv1 = expit(zeta1)
    pv2 = expit(zeta2)*expit(-zeta1)
    pv3 = expit(-zeta1)*expit(-zeta2)

    one_z_h_minus = (1-z)*h[:-1]
    scale = np.sqrt(1/inv_var)*np.ones((T,d))
    scale = scale.reshape(-1,d,1)

    V1 = -np.ones((T,d,1))
    mu1 = one_z_h_minus+z*V1

    V2 = np.ones((T,d,1))
    mu2 = one_z_h_minus+z*V2

    V3 = 1/alpha*fp
    mu3 = one_z_h_minus+z*V3

    

    pdf1 = norm.pdf(mu1,scale)
    pdf2 = norm.pdf(mu2,scale)
    pdf3 = norm.pdf(mu3,scale)
    
    norm_c = pdf1*pv1+pdf2*pv2+pdf3*pv3

    cond1 = pdf1*pv1/norm_c
    cond2 = pdf2*pv2/norm_c
    cond3 = pdf3*pv3/norm_c

    
    return np.concatenate((cond1,cond2,cond3), axis=2)


