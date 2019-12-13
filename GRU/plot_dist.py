import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import norm
import scipy.integrate as integrate
import sys


def get_binary():
    vec = []
    bn = [0,1]
    for z in bn:
        for r in bn:
            for v in bn:
                vec.append(np.array([z,r,v]))

    var = np.zeros((8,3))
    for i in range(0, len(vec)):
        var[i,:] = vec[i]
    return var


bin_vec = get_binary()




def bern_prob(p,x):
    return (p**x)*(1-p)**(1-x)




def get_likelihood(x, bin_vec, h_minus, var_h, Wz, Uz, bz,
                   Wr, Ur, br, Wp, Up, bp, u):

    pz = expit(Wz @ h_minus + Uz @ u + bz)
    pr = expit(Wr @ h_minus + Ur @ u + br)
    sum_L  = 0
    sum_mean = 0
    pv_mean = 0
    
    
    for j in range(0,len(bin_vec)):
        var = bin_vec[j,:]
        z = var[0]
        r = var[1]
        v = var[2]
        
        mean_h = (1-z)*h_minus+z*(2*v-1)

        pdf_h = norm.pdf(x, mean_h, np.sqrt(var_h))

        fp = Wp @ (r*h_minus) + Up @ u + bp
        pv = expit(2*fp)
        prob_z = bern_prob(pz,z)
        prob_r = bern_prob(pr,r)
        prob_v = bern_prob(pv,v)
        
        w = prob_z*prob_r*prob_v
        sum_L += pdf_h*w
        sum_mean += mean_h*w


    pv_mean = 0

    fp = Wp @ (0*h_minus) + Up @ u + bp
    pv = expit(2*fp)
    prob_r = bern_prob(pr,0)
    pv_mean += pv*prob_r

    fp = Wp @ (1*h_minus) + Up @ u + bp
    pv = expit(2*fp)
    prob_r = bern_prob(pr,1)
    pv_mean += pv*prob_r

    
    return sum_L, sum_mean, pv_mean, pz, pr






def y_prior(x, y, Wy, by, var_y):
    mean_y = Wy*x + by
    return norm.pdf(y, mean_y[0,0], np.sqrt(var_y))

def y_prior2(x, y, Wy, by, var_y):
    mean_y = Wy*x + by
    return norm.pdf(y, mean_y[0,:], np.sqrt(var_y))

def y_h_pdf(x, bin_vec, h_minus, var_h, Wz, Uz, bz,
                   Wr, Ur, br, Wp, Up, bp, u, y, Wy, by, var_y):
    like,_,_,_,_ = get_likelihood(x, bin_vec, h_minus,
                                  var_h, Wz, Uz,  bz, Wr, Ur, br,
                                  Wp, Up, bp, u)
    y_p = y_prior(x, y, Wy, by, var_y)
    
    #joint = like[0,0,0]*y_p
    joint = like[0,0,0]*y_p[0,0,0]
    return joint

def marginal_y(y, bin_vec, h_minus, var_h, Wz, Uz, bz,
                   Wr, Ur, br, Wp, Up, bp, u,Wy, by, var_y):
    marg = integrate.quad(y_h_pdf, -np.inf, np.inf,
                          args=(bin_vec, h_minus, var_h, Wz,
                                Uz, bz, Wr, Ur, br, Wp, Up,
                                bp, u, y, Wy, by, var_y) )
    return marg[0]


def int_y(bin_vec, h_minus, var_h, Wz, Uz, bz,
                   Wr, Ur, br, Wp, Up, bp, u,Wy, by, var_y):
    y_int = integrate.quad(marginal_y, -np.inf, np.inf,
                           args = (bin_vec, h_minus, var_h, Wz, Uz, bz,
                                   Wr, Ur, br, Wp, Up, bp, u, Wy, by, var_y))
    return y_int
