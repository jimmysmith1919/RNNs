import numpy as np
import sys


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
    

def log_prob_ht(ht, h_tmin, zt, vt, sig_h_inv,d):
    val = -1/2*(ht-((1-zt)*h_tmin+zt*(2*vt-1))).T @ sig_h_inv @ (
        ht-((1-zt)*h_tmin+zt*(2*vt-1))) 
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

def log_prob_v_gamma(vt, gamma_t, rt, h_tmin, Wp, Up, bp, ut)
