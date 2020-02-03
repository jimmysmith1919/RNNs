import numpy as np
from PG_int import pgpdf
import sys
from scipy.special import expit
import log_prob

def build_prec_x(inv_var, Sigma_y_inv, Wy,
                 Wp, pg_v, v, T, d, alpha, tau):
    prec = np.zeros((T*d,T*d))
    Sigma_inv = np.diag(inv_var)
    
    for t in range(0,T-1):
        #Diagonal (Block)
        prec[t*d:t*d+d,t*d:t*d+d] = Wy.T @ Sigma_y_inv @ Wy        
        prec[t*d:t*d+d,t*d:t*d+d]+=Sigma_inv
        
        vcat = np.argmax(v[t+1], axis=1)
        indeq3 = vcat ==2
        
        prec[t*d:t*d+d,t*d:t*d+d] += (4/alpha**2)*(Wp.T @
                                     (np.diag(indeq3)*Sigma_inv) @ Wp)
        prec[t*d:t*d+d,t*d:t*d+d] += (4/tau**2)*(Wp.T @ np.diag(pg_v[t+1,:,0]+
                                                        pg_v[t+1,:,1]) @ Wp)


        
        #Off-Diagonal
        
        #cross = (-2/alpha)*(Wp*indeq3).T @ Sigma_inv
        cross = (-2/alpha)*Wp.T @ (np.diag(indeq3)*Sigma_inv)  
        
        prec[t*d:t*d+d,t*d+d:t*d+2*d] = cross
        prec[t*d+d:t*d+2*d, t*d:t*d+d] = cross.T
        
    prec[(T-1)*d:(T-1)*d+d,(T-1)*d:(T-1)*d+d] = Sigma_inv + (Wy.T @ Sigma_y_inv
                                                            @ Wy)
    return prec



def build_prec_x_kalman(inv_var, Sigma_y_inv, Wy, 
                        Wp, pg_v, v, T, d, alpha, tau):
    
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
        vcat = np.argmax(v[t+1], axis=1)
        indeq3 = vcat ==2
        
        J_dyn_11[t] += (4/tau**2)*(Wp.T @ np.diag(pg_v[t+1,:,0]+
                                                  pg_v[t+1,:,1]) @ Wp)
        J_dyn_11[t] += (4/alpha**2)*(Wp.T @
                                     (np.diag(indeq3)*Sigma_inv) @ Wp)
            
        #Off-Diagonal    
        cross = -2/alpha*Wp.T @ (np.diag(indeq3)*Sigma_inv)
        J_dyn_21[t] += cross.T
        
    
    return J_dyn_11, J_dyn_22, J_dyn_21, J_obs, J_ini



def build_prec_muT(x0, u, inv_var, y, Sigma_y_inv, Wy, by,
                   v, Wp, Up, bp, pg_v, T, d, alpha, tau):


    vcat = np.argmax(v, axis=2)
    ind1 = vcat == 0
    ind1 = ind1.reshape(T,d,1)
    ind2 = vcat == 1
    ind2 = ind2.reshape(T,d,1)
    ind3 = vcat == 2
    ind3 = ind3.reshape(T,d,1)
    ones = np.ones((T,d,1))
        
    val = np.zeros((T*d,1))
    val[:,0] += (Wy.T @ Sigma_y_inv @ (y-by)).ravel()    

    ###initial cross terms, None assuming h0=0
    #val[:d,0] +=  inv_var*((1-z[0,:,0])*x0)
    #rW0 = Wp*r[0,:,0]
    #val[:d,0] += 1/alpha * x0.T @ (zI3[0,:,0]*rW0).T @ np.diag(inv_var)
    ###
    
    sum_terms = -ones+2*ind2+ind3    
    val[:,0] += (sum_terms[:,:,0]*inv_var).ravel()
    Uub = Up @ u + bp
    val[:,0] += (ind3[:,:,0]*2/alpha*Uub[:,:,0]*inv_var).ravel()

    
    #rW = r[1:,:,:].transpose(0,2,1)*Wp    
    #rWT = rW.transpose(0,2,1)

    indgreat2 = ind2+ind3
    Uub = Up @ u[1:,:,:]+bp
    
    val[:(T-1)*d,0] +=  -1/tau*(2*Wp.T @ (ind1[1:,:,:]-1/2)).ravel()
    
        
    val[:(T-1)*d,0] += 1/tau*(2*Wp.T @
                              (ind2[1:,:,:]-indgreat2[1:,:,:]/2)).ravel()

    
    val[:(T-1)*d,0] +=  -(1/tau**2)*(2*Wp.T @ (
                                  pg_v[1:,:,0].reshape(T-1,d,1)*(2*Uub+alpha)+
                                  pg_v[1:,:,1].reshape(T-1,d,1)*(2*Uub-alpha)
                                          )  ).ravel()
    
    val[:(T-1)*d,0]+= ( (-4/alpha**2)*( Wp.T*ind3[1:].reshape(T-1,1,d) ) @ (
        np.diag(inv_var) ) @ (Uub*ind3[1:]) ).ravel()
    
    return  val

def build_prec_muT_kalman(x0, u, inv_var, y, Sigma_y_inv, Wy, by,
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


    ones = np.ones((T,d,1))
    indgreat2 = ind2+ind3
    Uub = Up @ u[1:,:,:]+bp
    
    Uub0 = (Up @ u[0,:,:] +bp).reshape(1,d,1)
    
    sum_terms = -ones+2*ind2+ind3
    h_ini = inv_var*( sum_terms[0,:,0]+ind3[0,:,0]*2/alpha*Uub0[0,:,0] )
        
    h_dyn_2 = np.zeros((T-1,d))   
    h_dyn_2 += sum_terms[1:,:,0]*inv_var
    h_dyn_2 += ind3[1:,:,0]*2/alpha*Uub[:,:,0]*inv_var
    
    h_dyn_1 = np.zeros((T-1,d))
    h_dyn_1 += ((-4/alpha**2)*( Wp.T*ind3[1:].reshape(T-1,1,d) ) @ (
                np.diag(inv_var) ) @ (Uub*ind3[1:]) )[:,:,0]
    #####
    
 
    h_dyn_1 += (-1/tau*(2*Wp.T @ (ind1[1:,:,:]-1/2)))[:,:,0]
    h_dyn_1 += (1/tau*(2*Wp.T @ (ind2[1:,:,:]-indgreat2[1:,:,:]/2)))[:,:,0]
    h_dyn_1 += (-(1/tau**2)*(2*Wp.T @ (
                                  pg_v[1:,:,0].reshape(T-1,d,1)*(2*Uub+alpha)+
                                  pg_v[1:,:,1].reshape(T-1,d,1)*(2*Uub-alpha)
                                          )  ) )[:,:,0]    

    return  h_dyn_1, h_dyn_2, h_obs, h_ini



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





def build_v_param(h,u,Wp,Up, bp, inv_var,T,d, alpha, tau):
    fp = 2*(Wp @ h[:-1,:,:] + Up @ u + bp)
    
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
        log_pmf.append(log_prob.h_prior_logpdf(h,v,fp,inv_var,T,d, alpha))

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
