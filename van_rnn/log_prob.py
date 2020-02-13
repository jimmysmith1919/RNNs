import numpy as np
import sys
import PG_int as PG
from scipy.stats import bernoulli as bern
from scipy.stats import multivariate_normal as MVN
from scipy.stats import norm
from scipy.special import expit


def y_log_pdf(h, y, Wy, by, Sigma_y_inv,T, yd):
    mean = Wy @ h[1:] + by
    scale = np.sqrt(1/np.diagonal(Sigma_y_inv))*np.ones((T,yd))
    scale = scale.reshape(-1,yd,1)
    logpdf = norm.logpdf(y, mean, scale)
    return logpdf

def h_prior_logpdf(h, v,fp,inv_var,T,d, alpha):
    scale = np.sqrt(1/inv_var)*np.ones((T,d))
    scale = scale.reshape(-1,d,1)

    V1 = np.zeros((T,d,1))
    V2 = np.ones((T,d,1))
    V3 = 1/(2*alpha)*fp+1/2
    
    V = np.zeros((T,d))
    V += v[:,:,0]*V1[:,:,0]
    V += v[:,:,1]*V2[:,:,0]
    V += v[:,:,2]*V3[:,:,0]

    V = V.reshape(T,d,1)

    mu = 2*V-1
    logpdf = norm.logpdf(h[1:],mu,scale)
    return logpdf

def wbar_prior_log_pdf(W, var_w, mu_prior):

    scale = np.sqrt(var_w)
    logpdf = norm.logpdf(W, mu_prior, scale)
       
    return np.sum(logpdf)


'''
def z_prior_logpmf(z,h,W,U,b,u):
    f = W @ h[:-1,:,:] + U @ u + b
    mu = expit(f)
    return bern.logpmf(z,mu)
'''


def v_prior_logpmf(h,v,fp,T,d, alpha, tau):
    zeta1 = (-fp-alpha)/tau
    zeta2 = (fp-alpha)/tau

    pv1 = expit(zeta1)
    pv2 = np.exp(np.log(expit(zeta2))+np.log(expit(-zeta1)))
    pv3 = 1-(pv1+pv2)

    
    pv = np.zeros((T,d))
    pv += v[:,:,0]*pv1[:,:,0]
    pv += v[:,:,1]*pv2[:,:,0]
    pv += v[:,:,2]*pv3[:,:,0]
    
    pv = pv.reshape(T,d,1)    
    return np.log(pv)


def log_joint_no_weights(T, d, yd, u, y, h, inv_var, v,
                         Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau):
    log_like = 0
    
    fp = 2*(Wp @ h[:-1,:,:] + Up @ u + bp)

    log_like += h_prior_logpdf(h,v,fp,inv_var,T,d, alpha)    
    log_like += v_prior_logpmf(h,v,fp,T,d, alpha, tau)
    sum_log_like = np.sum(log_like)
    sum_log_like += np.sum(y_log_pdf(h,y,Wy, by, Sigma_y_inv,T, yd))
    return sum_log_like


def van_log_joint_no_weights(T, d, u, h, inv_var, v,
                         Wp, Up, bp,  alpha, tau):
    #for vanilla RNN no outptus
    log_like = 0
    
    fp = 2*(Wp @ h[:-1,:,:] + Up @ u + bp)

    log_like += h_prior_logpdf(h,v,fp,inv_var,T,d, alpha)    
    log_like += v_prior_logpmf(h,v,fp,T,d, alpha, tau)
    #sum_log_like = np.sum(log_like)
    
    return log_like

def log_joint_weights(T, d, yd, u, y, h, inv_var, v,
                      Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau,
                      Wp_bar, Wy_bar, Sigma_theta, Sigma_y_theta,
                      Wp_mu_prior, Wy_mu_prior):
    log_like = 0
    
    
    fp = 2*(Wp @ h[:-1,:,:] + Up @ u + bp)

    
    log_like += h_prior_logpdf(h,v,fp,inv_var,T,d, alpha)
    log_like += v_prior_logpmf(h,v,fp,T,d, alpha, tau)
    sum_log_like = np.sum(log_like)
    #sum_log_like += np.sum(y_log_pdf(h,y,Wy, by, Sigma_y_inv,T, yd))
    
    sum_log_like += wbar_prior_log_pdf(Wp_bar, Sigma_theta, Wp_mu_prior)
    
    #sum_log_like += wbar_prior_log_pdf(Wy_bar, Sigma_y_theta, Wy_mu_prior)

    return sum_log_like




def log_joint_pg_no_weights(T, d, yd, u, y, h, inv_var, v,
                            Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau,
                            gamma):
    log_like = 0
    
    fp = 2*(Wp @ h[:-1,:,:] + Up @ u + bp)

    
    log_like += h_prior_logpdf(h,v,fp,inv_var,T,d, alpha)
    log_like += log_prior_v_gamma_no_logpg(v, gamma, fp, alpha, tau, T,d)
    
    sum_log_like = np.sum(log_like)
    sum_log_like += np.sum(y_log_pdf(h,y,Wy, by, Sigma_y_inv,T, yd))
    
    return sum_log_like

def log_joint_pg_weights(T, d, yd, u, y, h, inv_var, v, 
                         Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau,
                         gamma,
                         Wp_bar, Wy_bar, Sigma_theta, Sigma_y_theta,
                          Wp_mu_prior, Wy_mu_prior):
    log_like = 0
    
    fp = 2*(Wp @ h[:-1,:,:] + Up @ u + bp)

    
    log_like += h_prior_logpdf(h,v,fp,inv_var,T,d, alpha)
    log_like += log_prior_v_gamma_no_logpg(v, gamma, fp, alpha, tau, T,d)
    sum_log_like = np.sum(log_like)
    sum_log_like += np.sum(y_log_pdf(h,y,Wy, by, Sigma_y_inv,T, yd))
    

    sum_log_like += wbar_prior_log_pdf(Wp_bar, Sigma_theta, Wp_mu_prior)
    sum_log_like += wbar_prior_log_pdf(Wy_bar, Sigma_y_theta, Wy_mu_prior)
    
    
    return sum_log_like


def log_joint_pg_gammalog_no_weights(T, d, yd, u, y, h, inv_var, v,
                                Wp, Up, bp, Wy, by, Sigma_y_inv, alpha, tau,
                                gamma, ind_great, zeta):
    log_like = 0
    
    fp = 2*(Wp @ h[:-1,:,:] + Up @ u + bp)

    
    log_like += h_prior_logpdf(h,v,fp,inv_var,T,d, alpha)
    
    val, plus_sum = log_prior_v_gamma(v, gamma, fp, alpha, tau, T,d,
                                  ind_great, zeta)
    log_like += val
    
    sum_log_like = np.sum(log_like)
    sum_log_like += np.sum(y_log_pdf(h,y,Wy, by, Sigma_y_inv,T, yd))
    sum_log_like += plus_sum
    return sum_log_like


def log_cond_v(v, cond_v):
    vcondv = v*cond_v
    check = vcondv != 0
    return np.sum(np.log(vcondv[check]))

def log_cond_z(z, Ez):
    log_z = bern.logpmf(z,Ez)
    return np.sum(log_z)

def loglike_normal(x, mean, covar):
    return MVN.logpdf(x, mean, covar)

def log_cond_Wbar(W_bar, W_mu, W_covar, d):
    log_like = 0
    for j in range(0,d):
        log_like += MVN.logpdf(W_bar[j,:], W_mu[j,:,0], W_covar[j])
    
    return log_like
    



'''
def log_prior_z_omega(z, omega, h, W, U, b, u, T,d):
    val = np.zeros((T,d,1))
    val += np.log(1/2)

    f = W @ h[:-1] + U @ u + b

    val += (z-1/2)*f
    val += -1/2*omega*(f**2)


    pdf_omega = PG.pgpdf(omega, 1, 0)
    zeros = np.zeros((T,d,1))
    check = np.isclose(pdf_omega, zeros, atol=1e-10)
    
    if np.sum(check)>0:
        print('pdf_omega element = 0, returning -inf')
        return -np.inf*np.ones((T,d,1))

    logpdf = np.log(pdf_omega)

    val += logpdf
    
    return val
'''

def log_prior_v_gamma_no_logpg(v, gamma, fp, alpha, tau, T,d):
    v_cat = np.argmax(v, axis=2).reshape(T,d,1)
    zeta1 = (-fp-alpha)/tau
    zeta2 = (fp-alpha)/tau
    zeta = [zeta1, zeta2]

    val = np.zeros((T,d,1))
    for k in range(0,2):
        ind_great = v_cat >= k
        ind_eq = v_cat == k

        val += np.log(1/(2**(ind_great)))
        val += (ind_eq-ind_great/2)*zeta[k]

        gamma_k = gamma[:,:,k].reshape(T,d,1)
        val += -1/2*(gamma_k)*(zeta[k]**2)
    return val


def log_prior_v_gamma(v, gamma, fp, alpha, tau, T,d, ind_great, zeta_flat):
    v_cat = np.argmax(v, axis=2).reshape(T,d,1)
    zeta1 = (-fp-alpha)/tau
    zeta2 = (fp-alpha)/tau
    zeta = [zeta1, zeta2]
    
    
    val = np.zeros((T,d,1))
    for k in range(0,2):
        ind_greatk = v_cat >= k
        ind_eq = v_cat == k

        val += np.log(1/(2**(ind_greatk)))
        val += (ind_eq-ind_greatk/2)*zeta[k]

        gamma_k = gamma[:,:,k].reshape(T,d,1)
        val += -1/2*(gamma_k)*(zeta[k]**2)

    gamma = np.concatenate((gamma[:,:,0].ravel(order='A'),
                                gamma[:,:,1].ravel(order='A')))
    '''
    ### For checking index 0
    l = len(ind_great)
    l = int(l/2)
    ind_greatnew = ind_great[:l]
    gamma = gamma[:,:,0].ravel(order='A')
    plus_sum = np.sum(np.log(PG.pgpdf(gamma, ind_greatnew, 0)))
    ###
    '''
    plus_sum = np.sum(np.log(PG.pgpdf(gamma, ind_great, 0)))
    
    return val, plus_sum






###################################################

def log_prob_yt(yt, ht, Wy, by, sig_y_inv, yd):
    val = -1/2*(yt-(Wy @ ht + by)).T @ sig_y_inv @ (yt-(Wy @ ht + by))
    val += -np.log((2*np.pi)**(yd/2)*np.linalg.det(
        np.linalg.inv(sig_y_inv))**(1/2))
    return val

def yt_wrapper(y0,y1,y2, ht, Wy, by, sig_y_inv, yd):
    yt = np.zeros((3,1))
    yt[0] = y0
    yt[1] = y1
    yt[2] = y2
    val = np.exp(log_prob_yt(yt, ht, Wy, by, sig_y_inv, yd))
    return val[0][0]


def log_prob_ht(ht, h_tmin, zt, vt, rt, Wp, Up, bp, ut, sig_h_inv, d, alpha):
    
    V_pre1 = -1*np.ones(d)
    V_pre2 = np.ones(d)
    V_pre3 =  1/alpha*(Wp @ (rt*h_tmin)+ Up @ ut + bp)
    
    V = vt[:,0]*V_pre1
    V += vt[:,1]*V_pre2
    V += vt[:,2]*V_pre3[:,0]
    V = V.reshape(d,1)
    
    val = -1/2*(ht-((1-zt)*h_tmin+zt*V)).T @ sig_h_inv @ (
        ht-((1-zt)*h_tmin+zt*V))
    
    val += -np.log((2*np.pi)**(d/2)*np.linalg.det(
        np.linalg.inv(sig_h_inv))**(1/2))
    
    return val

def ht_wrapper(h0,h1,h2, h_tmin, zt, vt, sig_h_inv, d):
    ht = np.zeros((3,1))
    ht[0] = h0
    ht[1] = h1
    ht[2] = h2
    val = np.exp(log_prob_ht(ht, h_tmin, zt, vt, sig_h_inv,d))
    return val[0][0]
'''
def log_prob_v_gamma(d, vt, gamma_t, rt, h_tmin, Wp, Up, bp, ut):
    val = d*np.log(1/2)
    
    fp = Wp @ (rt*h_tmin) + Up @ ut + bp
    vt_half = vt-1/2
    
    val += np.dot(vt_half[:,0],2*fp[:,0])

    fp2_sq = (2*fp)**2 
    val += -1/2*np.dot(gamma_t[:,0], fp2_sq[:,0])


    pdf_gamma = PG.pgpdf(gamma_t, 1, 0)
    zeros = np.zeros((d,1))
    check =  np.isclose(pdf_gamma,zeros, atol=1e-10)
    if np.sum(check) > 0:
        return -np.inf
    logpdf = np.log(pdf_gamma)
    
    val += np.sum(logpdf)
    return val
'''

def log_prob_v_gamma(d, vt, gamma_t, rt, h_tmin, Wp, Up, bp, ut, alpha, tau):

    v_cat = np.argmax(vt, axis=1)
        
    
    fp = Wp @ (rt*h_tmin) + Up @ ut + bp
    zeta1 = (-fp-alpha)/tau
    zeta2 = (fp-alpha)/tau
    zeta = [zeta1, zeta2]
    
    val = 0

    for k in range(0,2):
        
        ind_great = v_cat >=k
        
        if np.sum(ind_great) < d:
            #Need to check
            print(-np.inf) 
            return -np.inf
        ind_eq = v_cat == k

        val += np.sum(np.log(1/2**ind_great))
        coeff1 = ind_eq-ind_great/2
        
        val +=  np.sum(coeff1*zeta[k][:,0])

        val += np.sum(-1/2*gamma_t[:,k]*(zeta[k][:,0])**2)

        
        pdf_gamma = PG.pgpdf(gamma_t[:,k], ind_great, 0)
        
        zeros = np.zeros((d))
        check =  np.isclose(pdf_gamma,zeros, atol=1e-10)
        
        if np.sum(check) > 0:
            return -np.inf
        logpdf = np.log(pdf_gamma)
        
    
        val += np.sum(logpdf)
    
    return val


def v_gamma_wrapper(g0, g1, d, vt, rt, h_tmin, Wp, Up, bp, ut):
    gamma_t = np.zeros((2,1))
    gamma_t[0] = g0
    gamma_t[1] = g1
    val = np.exp(log_prob_v_gamma(d, vt, gamma_t, rt, h_tmin, Wp, Up, bp, ut))
    return val


'''
def v_gamma_wrapper(g0, vt, d,rt, h_tmin, Wp, Up, bp, ut):
    gamma_t = np.zeros((1,1))
    gamma_t[0] = g0
    val = np.exp(log_prob_v_gamma(d, vt, gamma_t, rt, h_tmin, Wp, Up, bp, ut))
    return val
'''

def log_prob_z(d,zt,h_tmin, W, U, b, ut):
    f = W @ h_tmin + U @ ut + b
    mu = expit(f) 
    return np.sum(bern.logpmf(zt, mu))
    
    

def log_prob_z_omega(d, zt, omega_t, h_tmin, W, U, b, ut):
    val = d*np.log(1/2)
    f = W @ h_tmin + U @ ut + b
    zt_half = zt-1/2
    val += np.dot(zt_half[:,0],f[:,0])
    f_sq = f**2 
    val += -1/2*np.dot(omega_t[:,0], f_sq[:,0])

    pdf_omega = PG.pgpdf(omega_t, 1, 0)
    zeros = np.zeros((d,1))
    check =  np.isclose(pdf_omega,zeros, atol=1e-10)
    if np.sum(check) > 0:
        return -np.inf
    
    logpdf = np.log(pdf_omega)
    
    val += np.sum(logpdf)
    
    return val

def z_omega_wrapper(o0, o1, d, zt, h_tmin, W, U, b, ut):
    omega_t = np.zeros((2,1))
    omega_t[0] = o0
    omega_t[1] = o1
    val = np.exp(log_prob_z_omega(d, zt, omega_t, h_tmin, W, U, b, ut))
    return val

def log_prob_wbar_dT(Wd_T, sig_inv_wd, mu_prior_d, d, ud):
    val = -np.log((2*np.pi)**((d+ud+1)/2)*np.linalg.det(
        np.linalg.inv(sig_inv_wd))**(1/2))
    val += -1/2*((Wd_T-mu_prior_d).T @ sig_inv_wd @ (Wd_T-mu_prior_d))
    return val

def wbar_wrapper(w0,w1,w2,w3, sig_W_inv, mu_prior_d, d, ud):
    Wd_t = np.zeros((4,1))
    Wd_t[0] = w0
    Wd_t[1] = w1
    Wd_t[2] = w2
    Wd_t[3] = w3
    print(Wd_t)
    val = np.exp(log_prob_wbar_dT(Wd_t, sig_W_inv, mu_prior_d, d, ud))
    return val[0][0]

def marg_log_joint_t_no_weights(d, yd, ut, yt, ht, h_tmin,
                                sig_h_inv, zt, vt, rt,
                                Wz, Uz, bz, Wr, Ur, br,
                                Wp, Up, bp, Wy, by, sig_y_inv, alpha, tau):
    val = log_prob_yt(yt, ht, Wy, by, sig_y_inv, yd)
    val += log_prob_ht(ht, h_tmin, zt, vt, rt, Wp, Up, bp, ut, sig_h_inv,d, alpha)
    val += log_prob_z(d, zt, h_tmin, Wz, Uz, bz, ut)
    val += log_prob_z(d, rt, h_tmin, Wr, Ur, br, ut)
    val += log_prob_z(d, vt, h_tmin, 2*Wp, 2*Up, 2*bp, ut)
    return val


def log_joint_t_no_weights(d, yd, ut, yt, ht, h_tmin, sig_h_inv, zt, vt, rt,
                           gamma_t, omega_zt, omega_rt,
                           Wz, Uz, bz, Wr, Ur, br,
                           Wp, Up, bp, Wy, by, sig_y_inv, alpha, tau):
    val = log_prob_yt(yt, ht, Wy, by, sig_y_inv, yd)
    val += log_prob_ht(ht, h_tmin, zt, vt, rt, Wp, Up, bp, ut,sig_h_inv,d, alpha)
    val += log_prob_v_gamma(d, vt, gamma_t, rt, h_tmin, Wp, Up, bp, ut, alpha, tau)
    val += log_prob_z_omega(d, zt, omega_zt, h_tmin, Wz, Uz, bz, ut)
    val += log_prob_z_omega(d, rt, omega_rt, h_tmin, Wr, Ur, br, ut)
    return val




def full_log_joint_no_weights(T, d, yd, u, y, h, sig_h_inv, z, v, r,
                              gamma, omega_z, omega_r,
                              Wz, Uz, bz, Wr, Ur, br,
                              Wp, Up, bp, Wy, by, sig_y_inv, alpha, tau):    
    val = 0
    for t in range(0,T):
        val += log_joint_t_no_weights(d, yd, u[t], y[t], h[t+1], h[t],
                                 sig_h_inv, z[t], v[t], r[t],
                                 gamma[t], omega_z[t], omega_r[t],
                                 Wz, Uz, bz, Wr, Ur, br,
                                      Wp, Up, bp, Wy, by, sig_y_inv, alpha, tau)
    
    return val[0][0]

def marg_full_log_joint_no_weights(T, d, yd, u, y, h, sig_h_inv, z, v, r,
                                   Wz, Uz, bz, Wr, Ur, br,
                                   Wp, Up, bp, Wy, by, sig_y_inv, alpha, tau):    
    val = 0
    for t in range(0,T):
        val += marg_log_joint_t_no_weights(d, yd, u[t], y[t], h[t+1], h[t],
                                 sig_h_inv, z[t], v[t], r[t],
                                 Wz, Uz, bz, Wr, Ur, br,
                                           Wp, Up, bp, Wy, by, sig_y_inv, alpha, tau)
    
    return val[0][0]




def full_log_joint(T, d, yd, u, y, h, sig_h_inv, z, v, r,
                   gamma, omega_z, omega_r,
                   Wz, Uz, bz, Wr, Ur, br,
                   Wp, Up, bp, Wy, by, sig_y_inv,
                   sig_inv_wd, mu_prior, ud, W):    
    val = 0
    for t in range(0,T):
        val += log_joint_t_no_weights(d, yd, u[t], y[t], h[t+1], h[t],
                                 sig_h_inv, z[t], v[t], r[t],
                                 gamma[t], omega_z[t], omega_r[t],
                                 Wz, Uz, bz, Wr, Ur, br,
                                 Wp, Up, bp, Wy, by, sig_y_inv)
    
    for j in range(0,yd): #change yd to d for non y weights
        val += log_prob_wbar_dT(W[j,:], np.diag(sig_inv_wd[j,:]), mu_prior[j,:], yd, 0)
        
    return val[0][0]



def loglike_bern_t(xtd, mu_td):
    return np.sum(bern.logpmf(xtd, mu_td))

'''
def loglike_normal(x, mean, covar):
    return MVN.logpdf(x, mean, covar)
'''

def loglike_PG(gamma, phi, d):
    pdf_gamma = PG.pgpdf(gamma, 1, phi)
    zeros = np.zeros((d,1))
    check =  np.isclose(pdf_gamma,zeros, atol=1e-10)
    if np.sum(check) > 0:
        return -np.inf
    logpdf = np.log(pdf_gamma)
    return np.sum(logpdf)
