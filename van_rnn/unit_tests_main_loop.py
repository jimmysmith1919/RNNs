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
               Sigma_y_theta,Sigma_y, Wp_mu_prior,
               Wy_mu_prior, Wp, Up, bp, Wy, by,
               train_weights, u, y, h, log_check, alpha, tau):
    ###
    h_plot_samples = np.zeros((N,d))
    log_joint_vec = []
    ###
    h_samples_vec = np.zeros((N-N_burn-1,T,d))
    Wp_bar_samples_vec = np.zeros((N-N_burn-1,d,d+ud+1))
    Wy_bar_samples_vec = np.zeros((N-N_burn-1,yd,d+1))
    
    h_samples =0
    v_samples =0
    Wp_bar_samples = 0
    Wy_bar_samples = 0

    Wp_bar = Wp_mu_prior
    Wy_bar = Wy_mu_prior
    
    for k in range(0,N):
                
        
        #Update v
        cond_v = build.build_v_param(h,u,Wp,Up,bp,inv_var,T,d,alpha,tau)
        cond_v=cond_v.reshape(T*d,3)
        items = np.arange(3)
        v = update.vectorized_cat(cond_v, items)
        v = np.eye(3)[v]
        v = v.reshape(T,d,3)        

        
        ###### check v updates

        
        v = update.vectorized_cat(cond_v, items)
        v = np.eye(3)[v]
        log_cond_v1 = log_prob.log_cond_v(v,cond_v)
        v1 = v.reshape(T,d,3)        
    
        v = update.vectorized_cat(cond_v, items)
        v = np.eye(3)[v]
        log_cond_v2 = log_prob.log_cond_v(v,cond_v)
        v2 = v.reshape(T,d,3)        
        
        
        log_joint1 =log_prob.log_joint_no_weights(T, d, yd, u, y,
                                                  h, inv_var, v1, 
                                                  Wp, Up, bp, Wy, by,
                                                  Sigma_y_inv, alpha, tau)


        log_joint2 =log_prob.log_joint_no_weights(T, d, yd, u, y,
                                                  h, inv_var, v2,
                                                  Wp, Up, bp, Wy, by,
                                                  Sigma_y_inv, alpha, tau)

        print('v')
        print('log_joint1-log_joint2')
        print(log_joint1-log_joint2)
        print('log_cond1-log_cond2')
        print(log_cond_v1-log_cond_v2)
        #sys.exit()
        #Update v
        cond_v = build.build_v_param(h,u,Wp,Up,bp,inv_var,T,d,alpha,tau)
        cond_v=cond_v.reshape(T*d,3)
        items = np.arange(3)
        v = update.vectorized_cat(cond_v, items)
        v = np.eye(3)[v]
        v = v.reshape(T,d,3)        

        #######
        

        
        
        
        #update gamma pg
        gamma, zeta, ind_great = update.gamma_update(h, v, u,
                                                     Wp, Up, bp, T,
                                                     d, alpha, tau)
        
        
        ###### check pg_gamma update
        ##Make sure v index greater than 0, so no log like issues
        
        check = np.argwhere(v[:,:,0])
        zeros = np.zeros((check.shape[0],1),dtype=int)
        check1=np.concatenate((check,zeros), axis = 1)
        checkshape = check1.shape
        L = checkshape[0]
        
        for l in range(0,L):
            shape = check1[l]
            v[shape[0], shape[1], 0] = 0
            v[shape[0], shape[1], 1] = 1
        
        #print('v')
        #print(v)
        

        gamma, zeta, ind_great = update.gamma_update(h, v, u,
                                             Wp, Up, bp, T, d, alpha, tau)    

        ind_great = ind_great.astype(np.double)
        gamma = update.sample_post_gamma(ind_great, zeta,T,d)

        log_joint1 =log_prob.log_joint_pg_gammalog_no_weights(T, d, yd,
                            u, y, h, inv_var, v,
                            Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau,
                            gamma, ind_great,zeta)

        gamma = np.concatenate((gamma[:,:,0].ravel(order='A'),
                                gamma[:,:,1].ravel(order='A')))
        
        log_cond1 = np.sum(np.log(PG.pgpdf(gamma, ind_great, zeta)))
        
        

        gamma = update.sample_post_gamma(ind_great.astype(np.double),zeta,T,d)


        log_joint2 =log_prob.log_joint_pg_gammalog_no_weights(T, d, yd,
                            u, y, h, inv_var, v,
                            Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau,
                            gamma, ind_great, zeta)


        gamma = np.concatenate((gamma[:,:,0].ravel(order='A'),
                                gamma[:,:,1].ravel(order='A')))
        
        log_cond2 = np.sum(np.log(PG.pgpdf(gamma, ind_great, zeta)))



        print('gamma')
        print('log_joint1-log_joint2')
        print(log_joint1-log_joint2)
        print('log_cond1-log_cond2')
        print(log_cond1-log_cond2)
        gamma = update.sample_post_gamma(ind_great.astype(np.double),zeta,T,d)
        
        #######
        

        '''
        ###### check pg_gamma update
        ##check index  = 0
        
        v[:] = 0
        v[:,:,0] = 1
        

        gamma, zeta, ind_great = update.gamma_update(rh, v, u,
                                             Wp, Up, bp, T, d, alpha, tau)    

        
        ind_great = ind_great.astype(np.double)
        gamma = update.sample_post_gamma(ind_great, zeta,T,d)


        log_joint1 =log_prob.log_joint_pg_gammalog_no_weights(T, d, yd,
                            u, y, h, inv_var, z, v, r,
                            Wz, Uz, bz, Wr, Ur, br,
                            Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau,
                            omega_z, omega_r, gamma, ind_great,zeta)


        l = len(ind_great)
        l = int(l/2)
        ind_greatnew = ind_great[:l]
        zetanew = zeta[:l]
        gamma = gamma[:,:,0].ravel(order='A')
        
        
        
        log_cond1 = np.sum(np.log(PG.pgpdf(gamma, ind_greatnew, zetanew)))
        
        

        gamma = update.sample_post_gamma(ind_great.astype(np.double),zeta,T,d)


        log_joint2 =log_prob.log_joint_pg_gammalog_no_weights(T, d, yd,
                            u, y, h, inv_var, z, v, r,
                            Wz, Uz, bz, Wr, Ur, br,
                            Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau,
                             omega_z, omega_r, gamma, ind_great, zeta)


        gamma = gamma[:,:,0].ravel(order='A')
        
        
        log_cond2 = np.sum(np.log(PG.pgpdf(gamma, ind_greatnew, zetanew)))

        
        print('log_joint1-log_joint2')
        print(log_joint1-log_joint2)
        print('log_cond1-log_cond2')
        print(log_cond1-log_cond2)
        


        cond_v = build.build_v_param(h,z,rh,u,Wp,Up,bp,inv_var,T,d,alpha,tau)
        cond_v=cond_v.reshape(T*d,3)
        items = np.arange(3)
        v = update.vectorized_cat(cond_v, items)
        v = np.eye(3)[v]
        v = v.reshape(T,d,3)        

        #######
        '''

        
        
        
        #Update hs
        '''
        #Kalman sampling
        J_dyn_11,J_dyn_22,J_dyn_21,J_obs,J_ini = build.build_prec_x_kalman(
            inv_var, Sigma_y_inv, Wy,
            Wp, gamma, v, T, d, alpha, tau)
        h_dyn_1, h_dyn_2, h_obs, h_ini = build.build_prec_muT_kalman(
            h0, u, inv_var, y, Sigma_y_inv, Wy, by,
            v, Wp, Up, bp, gamma, T, d, alpha, tau)
        


        log_Z_obs = np.zeros(T)
        h =messages.kalman_info_sample(J_ini, h_ini, 0, J_dyn_11, J_dyn_21,
                                       J_dyn_22, h_dyn_1, h_dyn_2, 0,
                                       J_obs, h_obs, log_Z_obs)
        
        
        h = h.reshape(T,d,1)
        h = np.concatenate((h0.reshape(1,d,1), h), axis=0)
        
                                                                    
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
        v[0] = 0
        v[0,:,1]=1
        v[1] = 0
        v[1,0,2] = 1
        v[1,1,1] = 1
        '''
        #print('v')
        #print(v)
        
        
        #Full T*dXT*d matrix
        prec = build.build_prec_x(inv_var, Sigma_y_inv, Wy,
                                  Wp, gamma, v, T, d, alpha, tau)

        
        prec_muT = build.build_prec_muT(h0, u, inv_var, y, Sigma_y_inv, Wy, by,
                                        v, Wp, Up, bp, gamma, T, d, alpha, tau)
        
        
        mu, covar = update.update_normal_dim(prec,prec_muT)
        
        
        h = np.random.multivariate_normal(mu[:,0],covar)
        h = h.reshape(T,d,1)
        h = np.concatenate((h0.reshape(1,d,1), h), axis=0)
        

        
        ###### check h update
        h = np.random.multivariate_normal(mu[:,0],covar)
        log_cond1 = log_prob.loglike_normal(h,mu[:,0] , covar)
        h = h.reshape(T,d,1)
        h = np.concatenate((h0.reshape(1,d,1), h), axis=0)


        log_joint1 =log_prob.log_joint_pg_no_weights(T, d, yd, u, y,
                                                     h, inv_var,v,
                                                     Wp, Up, bp, Wy, by,
                                                     Sigma_y_inv, alpha, tau,
                                                     gamma)
        
        
        h = np.random.multivariate_normal(mu[:,0],covar)
        log_cond2 = log_prob.loglike_normal(h,mu[:,0] , covar)
        h = h.reshape(T,d,1)
        h = np.concatenate((h0.reshape(1,d,1), h), axis=0)

        
        log_joint2 =log_prob.log_joint_pg_no_weights(T, d, yd, u, y,
                                                     h, inv_var, v, 
                                                     Wp, Up, bp, Wy, by,
                                                     Sigma_y_inv, alpha, tau,
                                                     gamma)
        

        print('full h')
        print('log_joint1-log_joint2')
        print(log_joint1-log_joint2)
        print('log_cond1-log_cond2')
        print(log_cond1-log_cond2)

        #######
        

        

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

            

            ####
            #slight adjustment to x for tanh approx
            Wp_bar, Wp, Up, bp, Wp_mu, Wp_covar = update.Wbar_p_update(h,
                                                      gamma,v,2*x, 4*xxT,
                                                      1/Sigma_theta,
                                                      Wp_mu_prior,T,d,ud,
                                                      alpha, tau, inv_var)


            
            ####
            ###### check Wp update
        

            '''
            print('z')
            print(z)
            print('r')
            print(r)
            print('gamma')
            print(gamma)
            print('v')
            print(v)
            print('rx')
            print(rx)
            '''
            

            
            Wp_bar,Wp,Up,bp,W_mu, W_covar = update.Wbar_p_update(h, gamma,v,
                                                                 2*x, 4*xxT,
                                                    1/Sigma_theta,
                                                      Wp_mu_prior,T,d,ud,
                                                      alpha, tau, inv_var)
            

            log_cond1 = log_prob.log_cond_Wbar(Wp_bar, W_mu, W_covar, d)
            

            

            log_joint1 =log_prob.log_joint_pg_weights(T, d, yd, u, y, h,
                            inv_var,  v,
                            Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau,
                            gamma, 
                            Wp_bar, Wy_bar, Sigma_theta, Sigma_y_theta,
                            Wp_mu_prior, Wy_mu_prior)


            
            Wp_bar,Wp,Up,bp,W_mu, W_covar = update.Wbar_p_update(h, gamma,
                                                                 v,2*x, 4*xxT,
                                                    1/Sigma_theta,
                                                      Wp_mu_prior,T,d,ud,
                                                      alpha, tau, inv_var)


            log_cond2 = log_prob.log_cond_Wbar(Wp_bar, W_mu, W_covar, d)
            
            log_joint2 = log_prob.log_joint_pg_weights(T, d, yd, u, y, h,
                            inv_var,  v,
                            Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau,
                            gamma, 
                         Wp_bar, Wy_bar, Sigma_theta, Sigma_y_theta,
                         Wp_mu_prior, Wy_mu_prior)

            
        

            print('Wp_bar')
            print('log_joint1-log_joint2')
            print(log_joint1-log_joint2)
            print('log_cond1-log_cond2')
            print(log_cond1-log_cond2)
            
            #######
            
            
            
            

            
            #Update y weights
            x = np.concatenate((h[1:,:,:], np.ones((T,1,1))), axis=1)
            xxT = (x[...,None]*x[:,None,:]).reshape(T,d+1,d+1)
            Wy_bar, Wy, by, Wy_mu, Wy_covar = update.Wbar_y_update(x,xxT,y,
                                                Sigma_y_inv,
                                                 1/Sigma_y_theta, Wy_mu_prior,
                                                  T,d,yd)



            ###### check Wy update

            Wy_bar, Wy, by, W_mu, W_covar = update.Wbar_y_update(x,xxT,y, Sigma_y_inv,
                                                  1/Sigma_y_theta, Wy_mu_prior,
                                                  T,d,yd)
            
            

            

            log_cond1 = log_prob.log_cond_Wbar(Wy_bar, W_mu, W_covar, yd)
            

            

            log_joint1 =log_prob.log_joint_pg_weights(T, d, yd, u, y, h,
                            inv_var, v,
                            Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau,
                            gamma, 
                         Wp_bar, Wy_bar, Sigma_theta, Sigma_y_theta,
                        Wp_mu_prior, Wy_mu_prior)



            Wy_bar, Wy, by, W_mu, W_covar = update.Wbar_y_update(x,xxT,y, Sigma_y_inv,
                                                  1/Sigma_y_theta, Wy_mu_prior,
                                                  T,d,yd)
            
            

            log_cond2 = log_prob.log_cond_Wbar(Wy_bar, W_mu, W_covar, yd)
            

        
            log_joint2 = log_prob.log_joint_pg_weights(T, d, yd, u, y, h,
                            inv_var, v, 
                            Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau,
                            gamma,
                         Wp_bar, Wy_bar, Sigma_theta, Sigma_y_theta,
                        Wp_mu_prior, Wy_mu_prior)

            
        
            print('Wy_bar')
            print('log_joint1-log_joint2')
            print(log_joint1-log_joint2)
            print('log_cond1-log_cond2')
            print(log_cond1-log_cond2)
            #######
            
            
            

    
        
        if k > N_burn:
            h_samples_vec[k-N_burn-1] = h[1:].reshape(T,d)
            h_samples += h
            v_samples += v
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
                                                             v, 
                                                    Wp, Up, bp, Wy, by,
                                                    Sigma_y_inv, alpha, tau,
                                                    Wp_bar, Wy_bar,
                                                    Sigma_theta, Sigma_y_theta,
                                                    Wp_mu_prior, Wy_mu_prior) )

            print(log_joint_vec)
            sys.exit('exit main loop')
        
        ###
   
        
    return h_samples, v_samples, Wp_bar_samples, Wy_bar_samples, h_samples_vec,  Wp_bar_samples_vec, Wy_bar_samples_vec, h_plot_samples, log_joint_vec 
    

