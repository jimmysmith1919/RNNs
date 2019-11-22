import numpy as np
from scipy.special import expit

def GRU_step(h_0, u, Wi, Ui, bi, Wr, Ur, br, Wp, Up, bp, Wy, by):
    fi = Wi @ h_0 + Ui @ u + bi
    fr = Wr @ h_0 + Ur @ u + br
    i = expit(fi)
    r = expit(fr)
    
    fp = Wp @ (r*h_0) + Up @ u + bp
    p = np.tanh(fp)
    
    h = (1-i)*h_0 + i*p
    y = Wy @ h + by
    return i, r, p, h, y


def stoch_GRU_step_mean(h_0, u, Wi, Ui, bi, Wr, Ur, br, Wp, Up, bp, Wy, by):
    fi = Wi @ h_0 + Ui @ u + bi
    fr = Wr @ h_0 + Ur @ u + br
    Ezi = expit(fi)
    Ezr = expit(fr)

    fp = Wp @ (Ezr*h_0) + Up @ u + bp
    Ev = expit(2*fp)

    Eh = (1-Ezi)*h_0 + Ezi*(2*Ev-1)
    Ey = Wy @ Eh + by
    return Ezi, Ezr, Ev, Eh, Ey

def stoch_GRU_step(Sigma,h_0, u, Wi, Ui, bi, Wr, Ur, br, Wp, Up, bp,
                   Sigma_y, Wy, by):
    fi = Wi @ h_0 + Ui @ u + bi
    fr = Wr @ h_0 + Ur @ u + br

    Ezi = expit(fi)
    Ezr = expit(fr)
    
    i = np.random.binomial(1, Ezi)
    r = np.random.binomial(1, Ezr)
    
    fp = Wp @ (r*h_0) + Up @ u + bp
    Ev = expit(2*fp)
    v = np.random.binomial(1, Ev)

    Eh = (1-i)*h_0 + i*(2*v-1)
    h = np.random.multivariate_normal(Eh, Sigma)
    
    Ey = Wy @ h + by
    y= np.random.multivariate_normal(Ey, Sigma_y)
    return i, r, v, h, y
    

def generate_rec_inp(func, steps, h_0, u, 
             Wi, Ui, bi, 
             Wr, Ur, br, 
             Wp, Up, bp, 
             Wy, by):
    '''For recurrent inputs where previous y is next input u '''
    
    d = len(h_0)
    yd = len(u)
    i_vec = np.zeros((steps, d))
    r_vec = np.zeros((steps, d))
    p_vec = np.zeros((steps, d))
    h_vec = np.zeros((steps, d))
    y_vec = np.zeros((steps, yd))
    
    h = h_0
    for step in range(0,steps):
        i,r,p,h,y = func(h, u, 
                         Wi, Ui, bi, 
                         Wr, Ur, br, 
                         Wp, Up, bp, 
                         Wy, by)
        i_vec[step,:] = i
        r_vec[step,:] = r
        p_vec[step,:] = p
        h_vec[step,:] = h
        y_vec[step,:] = y
        
        u = y
    return i_vec, r_vec, p_vec, h_vec, y_vec
