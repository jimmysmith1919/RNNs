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

        '''
        ####
        ##v prior check
        fp = (Wp @ rh[:-1,:,:] + Up @ u + bp)
        pdf=0
        
        v0 = np.zeros((d,3))
        for i in range(0,3):
            v0[0,:]=0
            v0[0,i]=1
            for j in range(0,3):
                v0[1,:]=0
                v0[1,j]=1
                for k in range(0,3):
                    v0[2,:]=0
                    v0[2,k]=1
                    v[3]=v0
                    a = log_prob.v_prior_logpmf(h,v,fp,T,d, alpha, tau)    
                    b = np.sum(a[3])
                    pdf += np.exp(b)                    

        print(pdf)
        ####
        '''

        '''
        ###z and r prior check
        pdf=0
        
        z0 = np.zeros((d,1))
        for i in range(0,2):
            z0[0]=i
            for j in range(0,2):
                z0[1]=j
                for k in range(0,2):
                    z0[2]=k
                    
                    r[0]=z0
                    a = log_prob.z_prior_logpmf(r,h,Wr,Ur,br,u)    
                    b = np.sum(a[0])
                    pdf += np.exp(b)
                 
        print(pdf)
        '''
        
        #Update omega pgs
        omega_z = update.pg_update(1, h, u, Wz, Uz, bz, T, d)
        omega_r = update.pg_update(1, h, u, Wr, Ur, br, T, d)

                
        rh[:-1,:,:] = r*h[:-1,:,:]
        #Update v
        cond_v = build.build_v_param(h,z,rh,u,Wp,Up,bp,inv_var,T,d,alpha,tau)
        cond_v=cond_v.reshape(T*d,3)
        items = np.arange(3)
        v = update.vectorized_cat(cond_v, items)
        v = np.eye(3)[v]
        v = v.reshape(T,d,3)        

        '''
        ###### check v updates
        np.random.seed(12)
        v = update.vectorized_cat(cond_v, items)
        v = np.eye(3)[v]
        log_cond_v1 = log_prob.log_cond_v(v,cond_v)
        v1 = v.reshape(T,d,3)        
    
        v = update.vectorized_cat(cond_v, items)
        v = np.eye(3)[v]
        log_cond_v2 = log_prob.log_cond_v(v,cond_v)
        v2 = v.reshape(T,d,3)        
        
        
        log_joint1 =log_prob.log_joint_no_weights(T, d, yd, u, y,
                                                  h, inv_var, z, v1, r,
                                                  Wz, Uz, bz, Wr, Ur, br,
                                                  Wp, Up, bp, Wy, by,
                                                  Sigma_y_inv, alpha, tau)


        log_joint2 =log_prob.log_joint_no_weights(T, d, yd, u, y,
                                                  h, inv_var, z, v2, r,
                                                  Wz, Uz, bz, Wr, Ur, br,
                                                  Wp, Up, bp, Wy, by,
                                                  Sigma_y_inv, alpha, tau)

        print('log_joint1-log_joint2')
        print(log_joint1-log_joint2)
        print('log_cond1-log_cond2')
        print(log_cond_v1-log_cond_v2)
        sys.exit()
        #######
        '''

        
        
        
        #update gamma pg
        gamma = update.gamma_update(rh, v, u, Wp, Up, bp, T, d, alpha, tau)
        
        '''
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
        
        print('v')
        print(v)
        

        gamma, zeta, ind_great = update.gamma_update(rh, v, u,
                                             Wp, Up, bp, T, d, alpha, tau)    

        ind_great = ind_great.astype(np.double)
        gamma = update.sample_post_gamma(ind_great, zeta,T,d)

        log_joint1 =log_prob.log_joint_pg_gammalog_no_weights(T, d, yd,
                            u, y, h, inv_var, z, v, r,
                            Wz, Uz, bz, Wr, Ur, br,
                            Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau,
                            omega_z, omega_r, gamma, ind_great,zeta)

        gamma = np.concatenate((gamma[:,:,0].ravel(order='A'),
                                gamma[:,:,1].ravel(order='A')))
        
        log_cond1 = np.sum(np.log(PG.pgpdf(gamma, ind_great, zeta)))
        
        

        gamma = update.sample_post_gamma(ind_great.astype(np.double),zeta,T,d)


        log_joint2 =log_prob.log_joint_pg_gammalog_no_weights(T, d, yd,
                            u, y, h, inv_var, z, v, r,
                            Wz, Uz, bz, Wr, Ur, br,
                            Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau,
                             omega_z, omega_r, gamma, ind_great, zeta)


        gamma = np.concatenate((gamma[:,:,0].ravel(order='A'),
                                gamma[:,:,1].ravel(order='A')))
        
        log_cond2 = np.sum(np.log(PG.pgpdf(gamma, ind_great, zeta)))

        
        print('log_joint1-log_joint2')
        print(log_joint1-log_joint2)
        print('log_cond1-log_cond2')
        print(log_cond1-log_cond2)
        sys.exit()
        #######
        '''
        '''
        ###### check pg_gamma update
        ##check index greater 0, so no log like issues
        
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
        sys.exit()
        #######
        '''


        '''
        ###### check pg_z update
        omega_z, g = update.pg_update(1, h, u, Wz, Uz, bz, T, d)
        
        omega_z = update.sample_post_pg(1,g,T,d)
        log_cond1 = np.sum(np.log(PG.pgpdf(omega_z, 1, g)))
        

        log_joint1 =log_prob.log_joint_pg_no_weights(T, d, yd, u, y, h, inv_var, z, v, r,
                            Wz, Uz, bz, Wr, Ur, br,
                            Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau,
                            omega_z, omega_r, gamma)

        
        omega_z = update.sample_post_pg(1,g,T,d)
        log_cond2 = np.sum(np.log(PG.pgpdf(omega_z, 1, g)))
        

        log_joint2 =log_prob.log_joint_pg_no_weights(T, d, yd, u, y, h, inv_var, z, v, r,
                            Wz, Uz, bz, Wr, Ur, br,
                            Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau,
                            omega_z, omega_r, gamma)

        
        print('log_joint1-log_joint2')
        print(log_joint1-log_joint2)
        print('log_cond1-log_cond2')
        print(log_cond1-log_cond2)
        sys.exit()
        #######
        '''
        '''
        ###### check pg_r update
        omega_r, g = update.pg_update(1, h, u, Wr, Ur, br, T, d)
        
        omega_r = update.sample_post_pg(1,g,T,d)
        log_cond1 = np.sum(np.log(PG.pgpdf(omega_r, 1, g)))
        

        log_joint1 =log_prob.log_joint_pg_no_weights(T, d, yd, u, y, h, inv_var, z, v, r,
                            Wz, Uz, bz, Wr, Ur, br,
                            Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau,
                            omega_z, omega_r, gamma)

        
        omega_r = update.sample_post_pg(1,g,T,d)
        log_cond2 = np.sum(np.log(PG.pgpdf(omega_r, 1, g)))
        

        log_joint2 =log_prob.log_joint_pg_no_weights(T, d, yd, u, y, h, inv_var, z, v, r,
                            Wz, Uz, bz, Wr, Ur, br,
                            Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau,
                            omega_z, omega_r, gamma)

        
        print('log_joint1-log_joint2')
        print(log_joint1-log_joint2)
        print('log_cond1-log_cond2')
        print(log_cond1-log_cond2)
        sys.exit()
        #######
        '''

        




        
        #Update Zs

        '''
        fz = build.build_z_param_old(h,v,rh,inv_var.reshape(d,1), u,
                                 Wz, Uz, bz, Wp, Up, bp, d, alpha)
        Ez = update.update_bern(fz)
        print(Ez)
        '''
        
        Ez = build.build_z_param(h,v,rh,inv_var, u,
                                 Wz, Uz, bz, Wp, Up, bp, T, d, alpha)
        z = np.random.binomial(1,Ez, size=(T,d,1))

        '''
        ###### check z update
        #seed = np.random.randint(0, 1000000)
        np.random.seed(15)
       
        z = np.random.binomial(1,Ez, size=(T,d,1))
        log_cond1 = log_prob.log_cond_z(z,Ez)            

        
        log_joint1 =log_prob.log_joint_no_weights(T, d, yd, u, y,
                                                  h, inv_var, z, v, r,
                                                  Wz, Uz, bz, Wr, Ur, br,
                                                  Wp, Up, bp, Wy, by,
                                                  Sigma_y_inv, alpha, tau)
        
        z = np.random.binomial(1,Ez, size = (T,d,1))
        log_cond2 = log_prob.log_cond_z(z,Ez)        

        log_joint2 =log_prob.log_joint_no_weights(T, d, yd, u, y,
                                                  h, inv_var, z, v, r,
                                                  Wz, Uz, bz, Wr, Ur, br,
                                                  Wp, Up, bp, Wy, by,
                                                  Sigma_y_inv, alpha, tau)

        print('log_joint1-log_joint2')
        print(log_joint1-log_joint2)
        print('log_cond1-log_cond2')
        print(log_cond1-log_cond2)
        sys.exit()
        #######
        ''' 


        '''
        ####
        check = np.random.randint(0,d)
        print(check)
        ####
        '''
        #Update r's
        for j in range(0,d):
            Erd=build.build_rd_param(h,z,v,r,inv_var,u,
                                     Wp,Up,bp,Wr,Ur,br,alpha,tau,T,d,j)
            r[:,j,:] = np.random.binomial(1,Erd, size=(T,1))

            '''
            ###### check r updates
            if j == check:
        
                np.random.seed(14)
                r[:,j,:] = np.random.binomial(1,Erd)
                log_cond1 = log_prob.log_cond_z(r[:,j],Erd)
                
                
                
                
                log_joint1 =log_prob.log_joint_no_weights(T, d, yd, u, y,
                                                  h, inv_var, z, v, r,
                                                  Wz, Uz, bz, Wr, Ur, br,
                                                  Wp, Up, bp, Wy, by,
                                                  Sigma_y_inv, alpha, tau)

                r[:,j,:] = np.random.binomial(1,Erd)
                log_cond2 = log_prob.log_cond_z(r[:,j],Erd)
                
                

                log_joint2 =log_prob.log_joint_no_weights(T, d, yd, u, y,
                                                  h, inv_var, z, v, r,
                                                  Wz, Uz, bz, Wr, Ur, br,
                                                  Wp, Up, bp, Wy, by,
                                                  Sigma_y_inv, alpha, tau)

                print('log_joint1-log_joint2')
                print(log_joint1-log_joint2)
                print('log_cond1-log_cond2')
                print(log_cond1-log_cond2)
                sys.exit()
            #######
            '''
          

        
        
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

            
        '''
        ###### check h update
        h = np.random.multivariate_normal(mu[:,0],covar)
        log_cond1 = log_prob.loglike_normal(h,mu[:,0] , covar)
        h = h.reshape(T,d,1)
        h = np.concatenate((h0.reshape(1,d,1), h), axis=0)


        log_joint1 =log_prob.log_joint_pg_no_weights(T, d, yd, u, y,
                                                     h, inv_var, z, v, r,
                                                     Wz, Uz, bz, Wr, Ur, br,
                                                     Wp, Up, bp, Wy, by,
                                                     Sigma_y_inv, alpha, tau,
                                                     omega_z, omega_r, gamma)
        
        
        h = np.random.multivariate_normal(mu[:,0],covar)
        log_cond2 = log_prob.loglike_normal(h,mu[:,0] , covar)
        h = h.reshape(T,d,1)
        h = np.concatenate((h0.reshape(1,d,1), h), axis=0)

        
        log_joint2 =log_prob.log_joint_pg_no_weights(T, d, yd, u, y,
                                                     h, inv_var, z, v, r,
                                                     Wz, Uz, bz, Wr, Ur, br,
                                                     Wp, Up, bp, Wy, by,
                                                     Sigma_y_inv, alpha, tau,
                                                     omega_z, omega_r, gamma)
        

        print('log_joint1-log_joint2')
        print(log_joint1-log_joint2)
        print('log_cond1-log_cond2')
        print(log_cond1-log_cond2)
        sys.exit()
        #######
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

            
            Wz_bar, Wz, Uz, bz = update.Wbar_update(z,omega_z,x,xxT,
                                                    1/Sigma_theta,
                                                    Wz_mu_prior,T,d,ud)
            

            '''
            ###### check Wz update
            #h = np.random.multivariate_normal(mu[:,0],covar)
            #log_cond1 = log_prob.loglike_normal(h,mu[:,0] , covar)
            #h = h.reshape(T,d,1)
            #h = np.concatenate((h0.reshape(1,d,1), h), axis=0)

            Wz_bar,Wz,Uz,bz,W_mu, W_covar = update.Wbar_update(z,omega_z,x,xxT,
                                                    1/Sigma_theta,
                                                    Wz_mu_prior,T,d,ud)

            

            log_cond1 = log_prob.log_cond_Wbar(Wz_bar, W_mu, W_covar, d)
            

            

            log_joint1 =log_prob.log_joint_pg_weights(T, d, yd, u, y, h,
                            inv_var, z, v, r,
                            Wz, Uz, bz, Wr, Ur, br,
                            Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau,
                            omega_z, omega_r, gamma, Wz_bar, Wr_bar,
                         Wp_bar, Wy_bar, Sigma_theta, Sigma_y_theta,
                         Wz_mu_prior, Wr_mu_prior, Wp_mu_prior, Wy_mu_prior)


            
            Wz_bar,Wz,Uz,bz,W_mu, W_covar = update.Wbar_update(z,omega_z,x,xxT,
                                                    1/Sigma_theta,
                                                    Wz_mu_prior,T,d,ud)

            
            

            log_cond2 = log_prob.log_cond_Wbar(Wz_bar, W_mu, W_covar, d)
            

        
            log_joint2 = log_prob.log_joint_pg_weights(T, d, yd, u, y, h,
                            inv_var, z, v, r,
                            Wz, Uz, bz, Wr, Ur, br,
                            Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau,
                            omega_z, omega_r, gamma, Wz_bar, Wr_bar,
                         Wp_bar, Wy_bar, Sigma_theta, Sigma_y_theta,
                         Wz_mu_prior, Wr_mu_prior, Wp_mu_prior, Wy_mu_prior)

            
        

            print('log_joint1-log_joint2')
            print(log_joint1-log_joint2)
            print('log_cond1-log_cond2')
            print(log_cond1-log_cond2)
            sys.exit()
            #######
            '''

            
            Wr_bar, Wr, Ur, br = update.Wbar_update(r,omega_r,x,xxT,
                                                    1/Sigma_theta,
                                                    Wr_mu_prior,T,d,ud)


            '''
            ###### check Wz update

            Wr_bar,Wr,Ur,br,W_mu, W_covar = update.Wbar_update(r,omega_r,x,xxT,
                                                    1/Sigma_theta,
                                                    Wr_mu_prior,T,d,ud)

            

            log_cond1 = log_prob.log_cond_Wbar(Wr_bar, W_mu, W_covar, d)
            

            

            log_joint1 =log_prob.log_joint_pg_weights(T, d, yd, u, y, h,
                            inv_var, z, v, r,
                            Wz, Uz, bz, Wr, Ur, br,
                            Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau,
                            omega_z, omega_r, gamma, Wz_bar, Wr_bar,
                         Wp_bar, Wy_bar, Sigma_theta, Sigma_y_theta,
                         Wz_mu_prior, Wr_mu_prior, Wp_mu_prior, Wy_mu_prior)


            
            Wr_bar,Wr,Ur,br,W_mu, W_covar = update.Wbar_update(r,omega_r,x,xxT,
                                                    1/Sigma_theta,
                                                    Wr_mu_prior,T,d,ud)

            
            

            log_cond2 = log_prob.log_cond_Wbar(Wr_bar, W_mu, W_covar, d)
            

        
            log_joint2 = log_prob.log_joint_pg_weights(T, d, yd, u, y, h,
                            inv_var, z, v, r,
                            Wz, Uz, bz, Wr, Ur, br,
                            Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau,
                            omega_z, omega_r, gamma, Wz_bar, Wr_bar,
                         Wp_bar, Wy_bar, Sigma_theta, Sigma_y_theta,
                         Wz_mu_prior, Wr_mu_prior, Wp_mu_prior, Wy_mu_prior)

            
        

            print('log_joint1-log_joint2')
            print(log_joint1-log_joint2)
            print('log_cond1-log_cond2')
            print(log_cond1-log_cond2)
            sys.exit()
            #######
            '''



            ####
            #slight adjustment to x for tanh approx
            rx = np.concatenate((h[:-1,:,:]*r,u, np.ones((T,1,1))), axis=1)
            rxrxT = (rx[...,None]*rx[:,None,:]).reshape(T,d+ud+1,d+ud+1)
            #Wp_bar, Wp, Up, bp = update.Wbar_p_update(h, z, gamma,v,rx, rxrxT,
            #                                        1/Sigma_theta,
            #                                          Wp_mu_prior,T,d,ud,
            #                                          alpha, tau, inv_var)



            ####
            ###### check Wp update
        

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

            

            
            Wp_bar,Wp,Up,bp,W_mu, W_covar = update.Wbar_p_update(h, z, gamma,v,rx, rxrxT,
                                                    1/Sigma_theta,
                                                      Wp_mu_prior,T,d,ud,
                                                      alpha, tau, inv_var)
            

            log_cond1 = log_prob.log_cond_Wbar(Wp_bar, W_mu, W_covar, d)
            

            

            log_joint1 =log_prob.log_joint_pg_weights(T, d, yd, u, y, h,
                            inv_var, z, v, r,
                            Wz, Uz, bz, Wr, Ur, br,
                            Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau,
                            omega_z, omega_r, gamma, Wz_bar, Wr_bar,
                         Wp_bar, Wy_bar, Sigma_theta, Sigma_y_theta,
                         Wz_mu_prior, Wr_mu_prior, Wp_mu_prior, Wy_mu_prior)


            
            Wp_bar,Wp,Up,bp,W_mu, W_covar = update.Wbar_p_update(h,z, gamma,v,rx, rxrxT,
                                                    1/Sigma_theta,
                                                      Wp_mu_prior,T,d,ud,
                                                      alpha, tau, inv_var)



            

            log_cond2 = log_prob.log_cond_Wbar(Wp_bar, W_mu, W_covar, d)
            

        
            log_joint2 = log_prob.log_joint_pg_weights(T, d, yd, u, y, h,
                            inv_var, z, v, r,
                            Wz, Uz, bz, Wr, Ur, br,
                            Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau,
                            omega_z, omega_r, gamma, Wz_bar, Wr_bar,
                         Wp_bar, Wy_bar, Sigma_theta, Sigma_y_theta,
                         Wz_mu_prior, Wr_mu_prior, Wp_mu_prior, Wy_mu_prior)

            
        

            print('log_joint1-log_joint2')
            print(log_joint1-log_joint2)
            print('log_cond1-log_cond2')
            print(log_cond1-log_cond2)
            sys.exit()
            #######

            
            
            

            
            #Update y weights
            x = np.concatenate((h[1:,:,:], np.ones((T,1,1))), axis=1)
            xxT = (x[...,None]*x[:,None,:]).reshape(T,d+1,d+1)
            Wy_bar, Wy, by = update.Wbar_y_update(x,xxT,y, Sigma_y_inv,
                                                  1/Sigma_y_theta, Wy_mu_prior,
                                                  T,d,yd)



        print(Wz_bar)
        log_prob.wbar_prior_log_pdf(Wz_bar, 1/Sigma_theta, Wz_mu_prior)
        sys.exit()
        
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

                   
            log_joint_vec.append( log_prob.full_log_joint_no_weights(T, d, yd,
                                                                     u, y, h, sig_h_inv, z, v, r, gamma, omega_z, omega_r,
                              Wz, Uz, bz, Wr, Ur, br,
                                                                          Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau) )
            sys.exit('exit main loop')
        
        ###
   
        
    return h_samples, z_samples, r_samples, v_samples, Wz_bar_samples, Wr_bar_samples, Wp_bar_samples, Wy_bar_samples, h_samples_vec, Wz_bar_samples_vec, Wr_bar_samples_vec, Wp_bar_samples_vec, Wy_bar_samples_vec, h_plot_samples, log_joint_vec 
    
