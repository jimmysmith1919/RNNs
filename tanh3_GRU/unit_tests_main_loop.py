import numpy as np
from ssm import messages
import var_updates as update
import build_model as build
from scipy.special import expit
import sys
import  log_prob
import scipy.integrate as integrate


def gibbs_loop(N, N_burn, T, d,T_check, ud, yd, h0, inv_var, Sigma_y_inv,
               Sigma_theta,
               Sigma_y_theta,Sigma_y, Wz_mu_prior, Wr_mu_prior, Wp_mu_prior,
               Wy_mu_prior, Wz, Uz, bz, Wr, Ur, br, Wp, Up, bp, Wy, by,
               train_weights, u, y, h, r, rh, z):

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

        
        #Update pgs
        #omega_z = update.pg_update(1, h, u, Wz, Uz, bz, T, d)
        #omega_r = update.pg_update(1, h, u, Wr, Ur, br, T, d)

        omega_z = update.pg_update(1, h, u, Wz, Uz, bz, T, d)
        omega_r = update.pg_update(1, h, u, Wr, Ur, br, T, d)
    
        rh[:-1,:,:] = r*h[:-1,:,:]
        gamma = update.pg_update(1, rh, u, 2*Wp, 2*Up, 2*bp, T, d)

        
        
        #Update v
        fv = build.build_v_param(h,z,rh,u,Wp,Up,bp, inv_var, d)
        Ev = update.update_bern(fv)
        v  = np.random.binomial(1, Ev, size=(T,d,1))

        
        #####
        '''
        #log prob testing
        ####
        sig_h_inv = np.diag(inv_var)
        #val = log_prob.log_prob_ht(h[2], h[1], z[1], v[1], sig_h_inv, d)

        val = integrate.nquad(log_prob.ht_wrapper,
                              [[-np.inf,np.inf],[-np.inf,np.inf],
                               [-np.inf,np.inf] ], args = [h[1],
                                                           z[2], v[2],
                                                           sig_h_inv,
                                                           d])
        
        print('h')
        print(val)
        sys.exit()
        



        ####
        '''


        
        '''
        val = integrate.nquad(log_prob.v_gamma_wrapper,
                              [[-np.inf,np.inf],[-np.inf,np.inf],
                               [-np.inf,np.inf],[-np.inf,np.inf],
                               [-np.inf,np.inf],[-np.inf,np.inf] ],
                              args = [d, r[2], h[1], Wp, Up, bp, u[2]])
        '''
        
        '''
        Sigma_theta = np.diag(Sigma_theta[0,:])
        Sig_theta_inv = np.linalg.inv(Sigma_theta)
        
        val = integrate.nquad(log_prob.wbar_wrapper,
                              [[-np.inf,np.inf],[-np.inf,np.inf],
                               [-np.inf,np.inf],[-np.inf,np.inf]],
                              args = [Sig_theta_inv, Wz_mu_prior[0,:], d, ud])
        print(val)
        

        ####
        '''
        ###
        
        
        #Update Zs
        fz = build.build_z_param(h,v,inv_var.reshape(d,1), Wz, Uz, u, bz, d)
        Ez = update.update_bern(fz)
        z = np.random.binomial(1,Ez, size=(T,d,1))
        
        #Update r's
        ####
        #testing
        #l = 5
        #q = d-1 
        #Er = np.zeros((d,T,1))
        #####
        for j in range(0,d):
        #for j in range(0,q+1): #(testing)
            frd = build.build_rd_param(h,u,v,gamma,r,Wp,Up,bp,Wr,Ur,br,j)
            Erd = update.update_bern(frd)
            ##
            #Er[j] = Erd
            ##
            r[:,j,:] = np.random.binomial(1,Erd, size=(T,1))

           
        #Update hs
        
        J_dyn_11,J_dyn_22,J_dyn_21,J_obs,J_ini = build.build_prec_x_kalman(
            inv_var, Sigma_y_inv, Wy,
            Wz, omega_z, z, Wr, omega_r, r,
            Wp, gamma, v, T, d)
        h_dyn_1, h_dyn_2, h_obs, h_ini = build.build_prec_muT_kalman(
            h0, u, inv_var, y, Sigma_y_inv, Wy, by,
            z, omega_z, Wz, Uz, bz,
            r, omega_r, Wr, Ur, br,
            v, Wp, Up, bp, gamma, T, d)
        
        
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

        
        log_Z_obs = np.zeros(T)
        h =messages.kalman_info_sample(J_ini, h_ini, 0, J_dyn_11, J_dyn_21,
                                       J_dyn_22, h_dyn_1, h_dyn_2, 0,
                                       J_obs, h_obs, log_Z_obs)
        
        
        h = h.reshape(T,d,1)
        h = np.concatenate((h0.reshape(1,d,1), h), axis=0)
        

        '''
        prec = build.build_prec_x(inv_var, Sigma_y_inv, Wy,
                                  Wz, omega_z, z, Wr, omega_r, r,
                                  Wp, gamma, v, T, d)
        
        prec_muT = build.build_prec_muT(h0, u, inv_var, y, Sigma_y_inv, Wy, by,
                                        z, omega_z, Wz, Uz, bz,
                                        r, omega_r, Wr, Ur, br,
                                        v, Wp, Up, bp, gamma, T, d)
        
        
        mu, covar = update.update_normal_dim(prec,prec_muT)
        
        
        h = np.random.multivariate_normal(mu[:,0],covar)
        h = h.reshape(T,d,1)
        h = np.concatenate((h0.reshape(1,d,1), h), axis=0)
        '''
        

               
        '''
        #Sample y's, for Testing purposes                                      
        Ey = Wy @ h[1:] + by                                                   
        Sig_y_diag = np.diag(Sigma_y).reshape(yd,1)                            
        Sig_y_in = np.ones((T,yd,1))*Sig_y_diag                              
        y= np.random.normal(Ey, np.sqrt(Sig_y_in))   
        '''

        if train_weights == True:
        
             #Update Weights
            x = np.concatenate((h[:-1,:,:],u, np.ones((T,1,1))), axis=1)
            xxT = (x[...,None]*x[:,None,:]).reshape(T,d+ud+1,d+ud+1)

            Wz_bar, Wz, Uz, bz, Wz_covar, Wz_mu = update.Wbar_update(z,omega_z,x,xxT,
                                                    1/Sigma_theta,
                                                    Wz_mu_prior,T,d,ud)
        
            Wr_bar, Wr, Ur, br, Wr_covar, Wr_mu = update.Wbar_update(r,omega_r,x,xxT,
                                                    1/Sigma_theta,
                                                    Wr_mu_prior,T,d,ud)
        
            #slight adjustment to x for tanh approx
            rx = 2*np.concatenate((h[:-1,:,:]*r,u, np.ones((T,1,1))), axis=1)
            rxrxT = (rx[...,None]*rx[:,None,:]).reshape(T,d+ud+1,d+ud+1)
            Wp_bar, Wp, Up, bp, Wp_covar, Wp_mu = update.Wbar_update(v, gamma, rx, rxrxT,
                                                    1/Sigma_theta,
                                                    Wp_mu_prior,T,d,ud)
        
            #Update y weights
            x = np.concatenate((h[1:,:,:], np.ones((T,1,1))), axis=1)
            xxT = (x[...,None]*x[:,None,:]).reshape(T,d+1,d+1)
            Wy_bar, Wy, by, Wy_covar, Wy_mu = update.Wbar_y_update(x,xxT,y, Sigma_y_inv,
                                                  1/Sigma_y_theta, Wy_mu_prior,
                                                  T,d,yd)


        #Unit testing
        ####
        sig_h_inv = np.diag(inv_var)

        sig_inv_wd = 1/Sigma_y_theta
        mu_prior = Wy_mu_prior



        W = update.sample_weights(Wy_covar, Wy_mu, d, 0)
        Wy,Uy,by = update.extract_W_weights(W, d, 0)
        
        '''
        l = 1
        q = d-2
        '''
        '''
        gamma = update.sample_post_pg(1,g_gamma,T,d)
        val1_post = log_prob.loglike_PG(gamma, g_gamma, d)
        '''
        
        '''
        z[l,q,0] = 1
        val1_post = log_prob.loglike_bern_t(z[l,q,0], Ez[l,q,0])
        '''
        '''
        v[l,q,0] = 1
        val1_post = log_prob.loglike_bern_t(v[l,q,0], Ev[l,q,0])
        '''
        '''
        r[l,q,0]= 0
        val1_post = log_prob.loglike_bern_t(r[l,q,0], Erd[l])
        '''
        '''

        h = np.random.multivariate_normal(mu[:,0],covar)
        val1_post = log_prob.loglike_normal(h, mu[:,0], covar)
        

        h = h.reshape(T,d,1)
        h = np.concatenate((h0.reshape(1,d,1), h), axis=0)
        '''

        
        val1_post = 0
        for j in range(0,yd):
            val1_post += log_prob.loglike_normal(W[j,:], Wy_mu[j,:].reshape(d+0+1), Wy_covar[j,:,:])
        
        print('val1post')
        print(val1_post)
        
        
        val1_joint = log_prob.full_log_joint(T, d, yd, u, y, h,
                                        sig_h_inv, z, v, r,
                                        gamma, omega_z, omega_r,
                                        Wz, Uz, bz, Wr, Ur, br,
                                        Wp, Up, bp, Wy, by, Sigma_y_inv,
                                        sig_inv_wd, mu_prior, ud, W)
        
        
        '''
        val1_joint = log_prob.full_log_joint_no_weights(T, d, yd, u, y, h,
                                        sig_h_inv, z, v, r,
                                        gamma, omega_z, omega_r,
                                        Wz, Uz, bz, Wr, Ur, br,
                                        Wp, Up, bp, Wy, by, Sigma_y_inv)
        '''
        '''
        r[l,q,0] = 1 
        val2_post = log_prob.loglike_bern_t(r[l,q,0], Erd[l])
        '''
        '''
        v[l,q,0] = 0
        val2_post = log_prob.loglike_bern_t(v[l,q,0], Ev[l,q,0])
        '''
        '''
        z[l,q,0] = 0
        val2_post = log_prob.loglike_bern_t(z[l,q,0], Ez[l,q,0])
        '''

        '''
        h = np.random.multivariate_normal(mu[:,0],covar)
        val2_post = log_prob.loglike_normal(h, mu[:,0], covar)
        

        h = h.reshape(T,d,1)
        h = np.concatenate((h0.reshape(1,d,1), h), axis=0)
        '''
        '''
        gamma = update.sample_post_pg(1,g_gamma,T,d)
        val2_post = log_prob.loglike_PG(gamma, g_gamma, d)
        '''


        W = update.sample_weights(Wy_covar, Wy_mu, d, 0)
        Wy,Uy,by = update.extract_W_weights(W, d, 0)
        
        val2_post = 0
        for j in range(0,yd):
            val2_post += log_prob.loglike_normal(W[j,:], Wy_mu[j,:].reshape(d+0+1), Wy_covar[j,:,:])
        
        print(' ')
        print('val2post')
        print(val2_post)

        val2_joint = log_prob.full_log_joint(T, d, yd, u, y, h,
                                        sig_h_inv, z, v, r,
                                        gamma, omega_z, omega_r,
                                        Wz, Uz, bz, Wr, Ur, br,
                                        Wp, Up, bp, Wy, by, Sigma_y_inv,
                                        sig_inv_wd, mu_prior, ud, W)
        
        '''
        val2_joint = log_prob.full_log_joint_no_weights(T, d, yd, u, y, h,
                                        sig_h_inv, z, v, r,
                                        gamma, omega_z, omega_r,
                                        Wz, Uz, bz, Wr, Ur, br,
                                        Wp, Up, bp, Wy, by, Sigma_y_inv)
        '''

        print('val1post-val2post')
        print(val1_post-val2_post)
        print('val1joint-val2joint')
        print(val1_joint-val2_joint)
        
        
        sys.exit()
        
        #####


            
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

    return h_samples, z_samples, r_samples, v_samples, Wz_bar_samples, Wr_bar_samples, Wp_bar_samples, Wy_bar_samples, h_samples_vec, Wz_bar_samples_vec, Wr_bar_samples_vec, Wp_bar_samples_vec, Wy_bar_samples_vec 
    
