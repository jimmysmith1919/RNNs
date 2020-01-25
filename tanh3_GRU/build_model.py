import numpy as np
from PG_int import pgpdf
import sys
from scipy.special import expit
import log_prob

def build_prec_x(inv_var, Sigma_y_inv, Wy, Wz, pg_z, z, Wr,
                 pg_r, r, Wp, pg_v, v, T, d, alpha, tau):
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
        indeq3 = vcat ==2
        ind3z = indeq3*z[t+1,:,0]
        
        prec[t*d:t*d+d,t*d:t*d+d] += (1/alpha**2)*(rW.T @
                               (np.diag(ind3z)*Sigma_inv) @ rW)
        prec[t*d:t*d+d,t*d:t*d+d] += (1/tau**2)*(rW.T @ np.diag(pg_v[t+1,:,0]+
                                                       pg_v[t+1,:,1]) @ rW)

        #Off-Diagonal
        cross = -np.outer(1-z[t+1,:,0], np.ones(d))*Sigma_inv
        cross += -1/alpha*(rW.T*ind3z) @ Sigma_inv
        
        prec[t*d:t*d+d,t*d+d:t*d+2*d] = cross
        prec[t*d+d:t*d+2*d, t*d:t*d+d] = cross.T
        
    prec[(T-1)*d:(T-1)*d+d,(T-1)*d:(T-1)*d+d] = Sigma_inv + (Wy.T @ Sigma_y_inv
                                                            @ Wy)
    return prec



def build_prec_x_kalman(inv_var, Sigma_y_inv, Wy, Wz, pg_z, z, Wr,
                        pg_r, r, Wp, pg_v, v, T, d, alpha, tau):
    
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
        indeq3 = vcat ==2
        ind3z = indeq3*z[t+1,:,0]
        
        J_dyn_11[t] += (1/tau**2)*(rW.T @ np.diag(pg_v[t+1,:,0]+
                                                  pg_v[t+1,:,1]) @ rW)
        J_dyn_11[t] += (1/alpha**2)*(rW.T @
                               (np.diag(ind3z)*Sigma_inv) @ rW)
            
        #Off-Diagonal    
        cross = -np.outer(1-z[t+1,:,0], np.ones(d))*Sigma_inv
        cross += -1/alpha*(rW.T*ind3z) @ Sigma_inv
        J_dyn_21[t] += cross.T
        
    
    return J_dyn_11, J_dyn_22, J_dyn_21, J_obs, J_ini



def build_prec_muT(x0, u, inv_var, y, Sigma_y_inv, Wy, by,
                   z, pg_z, Wz, Uz, bz, r, pg_r, Wr, Ur, br,
                   v, Wp, Up, bp, pg_v, T, d, alpha, tau):


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
    
    
    ###initial cross terms, None assuming h0=0
    #val[:d,0] +=  inv_var*((1-z[0,:,0])*x0)
    #rW0 = Wp*r[0,:,0]
    #val[:d,0] += 1/alpha * x0.T @ (zI3[0,:,0]*rW0).T @ np.diag(inv_var)
    ###
    
    zV12 = zV1+zV2
    val[:,0] += (zV12[:,:,0]*inv_var).ravel()
    Uub = Up @ u + bp
    val[:,0] += (zI3[:,:,0]*1/alpha*Uub[:,:,0]*inv_var).ravel()
    
    
    val[:(T-1)*d,0] +=  (Wz.T @ (z[1:,:,:]-1/2)).ravel()
    val[:(T-1)*d,0] +=  (Wz.T @ -(pg_z[1:,:,:]*(Uz @ u[1:,:,:] + bz)) ).ravel()

    val[:(T-1)*d,0] +=  (Wr.T @ (r[1:,:,:]-1/2)).ravel()
    val[:(T-1)*d,0] +=  (Wr.T @ -(pg_r[1:,:,:]*(Ur @ u[1:,:,:] + br)) ).ravel()
    
    rW = r[1:,:,:].transpose(0,2,1)*Wp    
    rWT = rW.transpose(0,2,1)

    
    indgreat2 = ind2+ind3
    Uub = Up @ u[1:,:,:]+bp
    
    val[:(T-1)*d,0] +=  -1/tau*(rWT @ (ind1[1:,:,:]-1/2)).ravel()

    
    val[:(T-1)*d,0] += 1/tau*(rWT @ (ind2[1:,:,:]-indgreat2[1:,:,:]/2)).ravel()

    val[:(T-1)*d,0] +=  -(1/tau**2)*(rWT @ (
                                  pg_v[1:,:,0].reshape(T-1,d,1)*(Uub+alpha)+
                                  pg_v[1:,:,1].reshape(T-1,d,1)*(Uub-alpha)
                                          )  ).ravel()

    
    
    zzT = (z[...,None]*z[:,None,:]).reshape(T,d,d)

    

    val[:(T-1)*d,0]+= ( (-1/alpha**2)*( rWT*ind3[1:].reshape(T-1,1,d) ) @ (
        zzT[1:]*np.diag(inv_var) ) @ (Uub*ind3[1:]) ).ravel()
    

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
    
    Uub0 = (Up @ u[0,:,:] +bp).reshape(1,d,1)
    
    rW0 = Wp*r[0,:,0]

    h_ini = inv_var*( zV12[0,:,0]+zI3[0,:,0]*1/alpha*Uub0[0,:,0] )
    
    
    h_dyn_2 = np.zeros((T-1,d))   
    h_dyn_2 += zV12[1:,:,0]*inv_var
    h_dyn_2 += zI3[1:,:,0]*1/alpha*Uub[:,:,0]*inv_var


    '''
    ####I think this is h_dyn1, need to check

    
    zzT = (z[...,None]*z[:,None,:]).reshape(T,d,d)

    h_dyn_2 += ((-1/alpha**2)*( rWT*ind3[1:].reshape(T-1,1,d) ) @ (
        zzT[1:]*np.diag(inv_var) ) @ (Uub*ind3[1:]) )[:,:,0]
    #####
    '''


    h_dyn_1 = np.zeros((T-1,d))

    
    ####I think this is h_dyn1, need to check

    
    zzT = (z[...,None]*z[:,None,:]).reshape(T,d,d)

    h_dyn_1 += ((-1/alpha**2)*( rWT*ind3[1:].reshape(T-1,1,d) ) @ (
        zzT[1:]*np.diag(inv_var) ) @ (Uub*ind3[1:]) )[:,:,0]
    #####
    

    h_dyn_1 += (Wz.T @ (z[1:,:,:]-1/2))[:,:,0]
    h_dyn_1 += (Wz.T @ -(pg_z[1:,:,:]*(Uz @ u[1:,:,:] + bz)))[:,:,0]

    h_dyn_1 += (Wr.T @ (r[1:,:,:]-1/2))[:,:,0]
    h_dyn_1 += (Wr.T @ -(pg_r[1:,:,:]*(Ur @ u[1:,:,:] + br)) )[:,:,0]
 
    h_dyn_1 += (-1/tau*(rWT @ (ind1[1:,:,:]-1/2)))[:,:,0]
    h_dyn_1 += (1/tau*(rWT @ (ind2[1:,:,:]-indgreat2[1:,:,:]/2)))[:,:,0]
    h_dyn_1 += (-(1/tau**2)*(rWT @ (
                                  pg_v[1:,:,0].reshape(T-1,d,1)*(Uub+alpha)+
                                  pg_v[1:,:,1].reshape(T-1,d,1)*(Uub-alpha)
                                          )  ) )[:,:,0]

    return  h_dyn_1, h_dyn_2, h_obs, h_ini


'''
def build_z_param_old(h, v, rh, inv_var, u, W, U, b, Wp, Up, bp, d, alpha):
    h_minus_sq = (h[:-1,:,:])**2
    h_h_minus = h[:-1,:,:]*h[1:,:,:]
    
    fp = Wp @ rh[:-1,:,:] + Up @ u + bp
    
    V1 = v[:,:,0]*-1    
    V2 = v[:,:,1]
    V3 = v[:,:,2]*1/alpha*fp[:,:,0]
    
    V = V1+V2+V3
    V = V.reshape(-1,d,1)
    
    fz = W @ h[:-1,:,:] + U @ u + b
    
    fz += (-h_h_minus + h[1:,:,:]*V + 1/2*h_minus_sq - 1/2*(V**2))*inv_var
    
    return fz
'''
def build_z_param(h, v, rh, inv_var, u, W, U, b, Wp, Up, bp, T, d, alpha):

    log_terms = []

    
    fp = (Wp @ rh[:-1,:,:] + Up @ u + bp)
    for i in range(0,2):
        z = i*np.ones((T,d,1))
        
        logph = log_prob.h_prior_logpdf(h,z,v,fp,inv_var,T,d,alpha)   
        logpz = log_prob.z_prior_logpmf(z,h,W,U,b,u)
    
        log_terms.append(logph+logpz)

    log_den = np.log(np.exp(log_terms[0])+np.exp(log_terms[1]))
    log_cond = log_terms[1]-log_den
    
    cond = np.exp(log_cond)
    
    return cond




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
'''
'''
def build_v_param_old(h,z,rh,u,Wp,Up, bp, inv_var,T,d, alpha, tau):
    fp = (Wp @ rh[:-1,:,:] + Up @ u + bp)

    zeta1 = (-fp-alpha)/tau
    zeta2 = (fp-alpha)/tau

    pv1 = expit(zeta1)
    pv2 = expit(zeta2)*expit(-zeta1)
    pv3 = 1-(pv1+pv2)
    
    pdf = []
    for i in range(0,3):
        v = np.zeros((T,d,d))
        v[:,:,i] += 1
        pdf.append(log_prob.h_prior_pdf(h,z,v,fp,inv_var,T,d, alpha))

    pdf1 = pdf[0]
    pdf2 = pdf[1]
    pdf3 = pdf[2]
        
    norm_c = pdf1*pv1+pdf2*pv2+pdf3*pv3

    cond1 = pdf1*pv1/norm_c
    cond2 = pdf2*pv2/norm_c
    cond3 = pdf3*pv3/norm_c

    return np.concatenate((cond1,cond2,cond3), axis=2)
'''
def build_v_param(h,z,rh,u,Wp,Up, bp, inv_var,T,d, alpha, tau):
    fp = (Wp @ rh[:-1,:,:] + Up @ u + bp)
    
    log_pv = []
    for i in range(0,3):
        v = np.zeros((T,d,3))
        v[:,:,i] += 1
        log_pv.append(log_prob.v_prior_logpmf(h,v,fp,T,d, alpha, tau))
    
    log_pv1 = log_pv[0]
    log_pv2 = log_pv[1]
    log_pv3 = log_pv[2]

        
    log_pmf = []
    for i in range(0,3):
        v = np.zeros((T,d,3))
        v[:,:,i] += 1
        log_pmf.append(log_prob.h_prior_logpdf(h,z,v,fp,inv_var,T,d, alpha))

    log_pmf1 = log_pmf[0]
    log_pmf2 = log_pmf[1]
    log_pmf3 = log_pmf[2]


    logpost1 = log_pmf1+log_pv1
    logpost2 = log_pmf2+log_pv2
    logpost3 = log_pmf3+log_pv3
    
    lognorm_c = np.log(np.exp(logpost1)+np.exp(logpost2)+np.exp(logpost3))

    cond1 = np.exp( logpost1 - lognorm_c)
    cond2 = np.exp(logpost2 - lognorm_c )
    cond3 = 1-(cond1+cond2)
    
    return np.concatenate((cond1,cond2,cond3), axis=2)



def build_rd_param(h,z,v,r,inv_var,u,Wp,Up,bp,Wr,Ur,br,alpha,tau,T,d,dim):

    log_terms = []
    for i in range(0,2):
        r[:,dim]=i
        rh = np.zeros((T+1,d,1))
        rh[:-1,:,:] = r*h[:-1,:,:]
        fp = Wp @ rh[:-1,:,:] + Up @ u + bp
    
        logph = log_prob.h_prior_logpdf(h,z,v,fp,inv_var,T,d,alpha)
        logph = np.sum(logph, axis=1)
        
        logpv = log_prob.v_prior_logpmf(h,v,fp,T,d, alpha, tau)
        logpv = np.sum(logpv, axis=1)
        
        logpr = log_prob.z_prior_logpmf(r[:,dim],h,Wr[dim],Ur[dim],br[dim],u)
        
        log_terms.append(logph+logpv+logpr)

    log_den = np.log(np.exp(log_terms[0])+np.exp(log_terms[1]))
    log_cond = log_terms[1]-log_den
    
    cond = np.exp(log_cond)
    return cond
