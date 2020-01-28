import numpy as np
from ssm import messages
import var_updates as update
import build_model as build
from scipy.special import expit
import sys
import  log_prob
import scipy.integrate as integrate
import scipy.stats as stats
import PG_int as PG###can remove later


def gibbs_loop(N, N_burn, T, d,T_check, ud, yd, h0, inv_var, Sigma_y_inv,
               Sigma_theta,
               Sigma_y_theta,Sigma_y, Wz_mu_prior, Wr_mu_prior, Wp_mu_prior,
               Wy_mu_prior, Wz, Uz, bz, Wr, Ur, br, Wp, Up, bp, Wy, by,
               train_weights, u, y, h, r, rh, z, log_check, alpha, tau):
    ###
    h_plot_samples = np.zeros((N,d))
    log_joint_vec = []
    ###
    h_samples_vec = np.zeros((N-N_burn-1,T,d))
    Wz_bar_samples_vec = np.zeros((N-N_burn-1,d,d+ud+1))
    Wr_bar_samples_vec = np.zeros((N-N_burn-1,d,d+ud+1))
    Wp_bar_samples_vec = np.zeros((N-N_burn-1,d,d+ud+1))
    Wy_bar_samples_vec = np.zeros((N-N_burn-1,yd,d+1))
    
    h_samples =0
    z_samples =0
    r_samples =0
    v_samples =0
    Wz_bar_samples = 0
    Wr_bar_samples = 0
    Wp_bar_samples = 0
    Wy_bar_samples = 0

    Wz_bar = Wz_mu_prior
    Wr_bar = Wr_mu_prior
    Wp_bar = Wp_mu_prior
    Wy_bar = Wy_mu_prior
    
    for k in range(0,N):

        
        #Update omega pgs
        omega_z, _ = update.pg_update(1, h, u, Wz, Uz, bz, T, d)
        omega_r, _ = update.pg_update(1, h, u, Wr, Ur, br, T, d)

              
        rh[:-1,:,:] = r*h[:-1,:,:]
        #Update v
        cond_v = build.build_v_param(h,z,rh,u,Wp,Up,bp,inv_var,T,d,alpha,tau)
        cond_v=cond_v.reshape(T*d,3)
        items = np.arange(3)
        v = update.vectorized_cat(cond_v, items)
        v = np.eye(3)[v]
        v = v.reshape(T,d,3)                
        
        #update gamma pg
        gamma, _, _ = update.gamma_update(rh, v, u,
                                          Wp, Up, bp, T,
                                          d, alpha, tau)
    
        #Update Zs  
        Ez = build.build_z_param(h,v,rh,inv_var, u,
                                 Wz, Uz, bz, Wp, Up, bp, T, d, alpha)
        z = np.random.binomial(1,Ez, size=(T,d,1))

    
        #Update r's
        for j in range(0,d):
            Erd=build.build_rd_param(h,z,v,r,inv_var,u,
                                     Wp,Up,bp,Wr,Ur,br,alpha,tau,T,d,j)
            r[:,j,:] = np.random.binomial(1,Erd, size=(T,1))

        
        #Update hs
        
        #Kalman sampling
        J_dyn_11,J_dyn_22,J_dyn_21,J_obs,J_ini = build.build_prec_x_kalman(
            inv_var, Sigma_y_inv, Wy,
            Wz, omega_z, z, Wr, omega_r, r,
            Wp, gamma, v, T, d, alpha, tau)
        h_dyn_1, h_dyn_2, h_obs, h_ini = build.build_prec_muT_kalman(
            h0, u, inv_var, y, Sigma_y_inv, Wy, by,
            z, omega_z, Wz, Uz, bz,
            r, omega_r, Wr, Ur, br,
            v, Wp, Up, bp, gamma, T, d, alpha, tau)
        


        log_Z_obs = np.zeros(T)
        h =messages.kalman_info_sample(J_ini, h_ini, 0, J_dyn_11, J_dyn_21,
                                       J_dyn_22, h_dyn_1, h_dyn_2, 0,
                                       J_obs, h_obs, log_Z_obs)
        
        
        h = h.reshape(T,d,1)
        h = np.concatenate((h0.reshape(1,d,1), h), axis=0)
        
        '''                                                            
        #####                                                                  
        log_Z_obs = np.zeros(T)                                                
                                                                               
        _,smoothed_mus,smoothed_Sigmas,off=messages.kalman_info_smoother(J_ini,
                                                                       h_ini,  
                                                                       0,      
                         J_dyn_11, J_dyn_21, J_dyn_22, h_dyn_1, h_dyn_2, 0,    
                                       J_obs, h_obs, log_Z_obs)                
                                                                               
        print('smoothed_mus')                                                  
        print(smoothed_mus)                                                    
        print('smoothed_sigmas')                                               
        print(smoothed_Sigmas)                                                 
        print('off')                                                           
        print(np.round(off, 4))                                                
        #####                                                                  
        '''
        
        
        '''
        #Full T*dXT*d matrix
        prec = build.build_prec_x(inv_var, Sigma_y_inv, Wy,
                                  Wz, omega_z, z, Wr, omega_r, r,
                                  Wp, gamma, v, T, d, alpha, tau)

        
        prec_muT = build.build_prec_muT(h0, u, inv_var, y, Sigma_y_inv, Wy, by,
                                        z, omega_z, Wz, Uz, bz,
                                        r, omega_r, Wr, Ur, br,
                                        v, Wp, Up, bp, gamma, T, d, alpha, tau)
        
        
        mu, covar = update.update_normal_dim(prec,prec_muT)

        
        
        h = np.random.multivariate_normal(mu[:,0],covar)
        h = h.reshape(T,d,1)
        h = np.concatenate((h0.reshape(1,d,1), h), axis=0)
        '''
        

        
        ############
        ##Sample Y's, FOR GEWECKE TESTING ONLY
        Ey = Wy @ h[1:] + by                                                   
        Sig_y_diag = np.diag(Sigma_y).reshape(yd,1)                            
        Sig_y_in = np.ones((T,yd,1))*Sig_y_diag                              
        y= np.random.normal(Ey, np.sqrt(Sig_y_in))
        ################
        

        if train_weights == True:
        
            #Update Weights
            x = np.concatenate((h[:-1,:,:],u, np.ones((T,1,1))), axis=1)
            xxT = (x[...,None]*x[:,None,:]).reshape(T,d+ud+1,d+ud+1)

            
            Wz_bar, Wz, Uz, bz, _, _ = update.Wbar_update(z,
                                                          omega_z,
                                                          x,xxT, 1/Sigma_theta,
                                                          Wz_mu_prior,T,d,ud)
            
            Wr_bar, Wr, Ur, br, _, _ = update.Wbar_update(r,
                                                    omega_r,x,xxT,
                                                    1/Sigma_theta,
                                                    Wr_mu_prior,T,d,ud)

            ####
            #slight adjustment to x for tanh approx
            rx = np.concatenate((h[:-1,:,:]*r,u, np.ones((T,1,1))), axis=1)
            rxrxT = (rx[...,None]*rx[:,None,:]).reshape(T,d+ud+1,d+ud+1)
            Wp_bar, Wp, Up, bp, _, _ = update.Wbar_p_update(h, z,
                                                      gamma,v,rx, rxrxT,
                                                      1/Sigma_theta,
                                                      Wp_mu_prior,T,d,ud,
                                                      alpha, tau, inv_var)
        
            
            #Update y weights
            x = np.concatenate((h[1:,:,:], np.ones((T,1,1))), axis=1)
            xxT = (x[...,None]*x[:,None,:]).reshape(T,d+1,d+1)
            Wy_bar, Wy, by, _, _ = update.Wbar_y_update(x,xxT,y,
                                                Sigma_y_inv,
                                                 1/Sigma_y_theta, Wy_mu_prior,
                                                  T,d,yd)
    
        if k > N_burn:
            h_samples_vec[k-N_burn-1] = h[1:].reshape(T,d)
            h_samples += h
            z_samples += z
            r_samples += r
            v_samples += v
            Wz_bar_samples += Wz_bar
            Wz_bar_samples_vec[k-N_burn-1] = Wz_bar
            Wr_bar_samples += Wr_bar
            Wr_bar_samples_vec[k-N_burn-1] = Wr_bar
            Wp_bar_samples += Wp_bar
            Wp_bar_samples_vec[k-N_burn-1] = Wp_bar
            Wy_bar_samples += Wy_bar
            Wy_bar_samples_vec[k-N_burn-1] = Wy_bar
        print(k)
        ###
        h_plot_samples[k] = h[T_check,:,0]

        sig_h_inv = np.diag(inv_var)
        if k%log_check == 0:

                   
            log_joint_vec.append( log_prob.log_joint_weights(T, d, yd, u, y,
                                                             h, inv_var,
                                                             z, v, r,
                                                    Wz, Uz, bz, Wr, Ur, br,
                                                    Wp, Up, bp, Wy, by,
                                                    Sigma_y_inv, alpha, tau,
                                                    Wz_bar, Wr_bar,
                                                    Wp_bar, Wy_bar,
                                                    Sigma_theta, Sigma_y_theta,
                                                    Wz_mu_prior, Wr_mu_prior,
                                                    Wp_mu_prior, Wy_mu_prior) )
        ###
   
        
    return h_samples, z_samples, r_samples, v_samples, Wz_bar_samples, Wr_bar_samples, Wp_bar_samples, Wy_bar_samples, h_samples_vec, Wz_bar_samples_vec, Wr_bar_samples_vec, Wp_bar_samples_vec, Wy_bar_samples_vec, h_plot_samples, log_joint_vec 
    
