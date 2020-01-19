import numpy as np
import sys
import PG_int as PG
from scipy.stats import bernoulli as bern
from scipy.stats import multivariate_normal as MVN
from scipy.special import expit

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


def loglike_normal(x, mean, covar):
    return MVN.logpdf(x, mean, covar)

def loglike_PG(gamma, phi, d):
    pdf_gamma = PG.pgpdf(gamma, 1, phi)
    zeros = np.zeros((d,1))
    check =  np.isclose(pdf_gamma,zeros, atol=1e-10)
    if np.sum(check) > 0:
        return -np.inf
    logpdf = np.log(pdf_gamma)
    return np.sum(logpdf)
