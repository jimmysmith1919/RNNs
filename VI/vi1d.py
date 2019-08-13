import numpy as np
from scipy.special import expit


##1-d variational inference algorithm, NO y OUTPUTS ##

##Update q(h)##
def update_qh(T, prec_h, E_z, E_v, E_omega, W):
    prec_hat = np.zeros(T)
    prec_mu_hat = np.zeros(T)
    
    for t in range(0,T-1):
        sum1 = 0
        for i in range(0,4):
            sum1 += E_omega[i,t]*W[i]**2
        prec_hat[t] = prec_h+sum1

        for i in range(0,4):
            sum2 = (E_z[i,t]-.5)*W[i]
        #note E_z[3,t]=E_zo[t]
        prec_mu_hat[t] = ( E_z[3,t]*(2*E_v[t]-1) )*prec_h + sum2 

    prec_hat[T-1] = prec_h
    prec_mu_hat[T-1] = ( E_z[3,T-1]*(2*E_v[T-1]-1) )*prec_h
    return prec_hat, prec_mu_hat

def update_Eh(T, prec_hat, prec_mu_hat):
    Eh = np.zeros(T)
    Ehh = np.zeros(T)
    for t in range(0,T):
        Eh[t] = prec_mu_hat[t]/prec_hat[t]
        Ehh[t] = Eh[t]**2+1/prec_hat[t]
    return Eh, Ehh

def update_qc(T, prec_c, E_gamma, E_z, E_v):
    prec_hat = np.zeros((T,T))
    prec_mu_hat = np.zeros(T)

    prec_hat[0,0] = prec_c+4*E_gamma[0]
    prec_mu_hat[0] = ( E_z[0,0]*(2*E_z[2,0]-1) )*prec_c + 2*E_v[0]-1
    for t in range(1,T):
        prec_hat[t,t] = prec_c+4*E_gamma[t]
        prec_hat[t-1,t] = -E_z[1,t]*prec_c
        prec_hat[t,t-1] = -E_z[1,t]*prec_c
        prec_mu_hat[t] = ( E_z[0,t]*(2*E_z[2,t]-1) )*prec_c + 2*E_v[t]-1
    
    covar = np.linalg.inv(prec_hat)
    Ec = covar @ prec_mu_hat
    Ecc = np.outer(Ec,Ec)+covar
    return Ec, Ecc

def update_qv(T, prec_h, Ec, Ez, Eh, Ezozo):
    prec_hat = np.zeros(T)
    prec_mu_hat = np.zeros(T)
    
    for t in range(0,T):
        prec_hat[t] = 4*Ezozo[t]*prec_h
        prec_mu_hat[t] = 2*Ec[t]+2*Ezozo[t]*prec_h+2*Ez[3,t]*Eh[t]*prec_h
    
    covar = np.linalg.inv(np.diag(prec_hat))
    Ev = covar @ prec_mu_hat
    Evv = np.outer(Ev, Ev)+covar
    return Ev, Evv
    


##Random Weights##                                                          
np.random.seed(0)

Wi = np.random.randn(1)
Wf = np.random.randn(1)
Wp = np.random.randn(1)
Wo = np.random.randn(1)
W = np.concatenate((Wi,Wf,Wp,Wo))

Ui = np.random.randn(1)
Uf = np.random.randn(1)
Up = np.random.randn(1)
Uo = np.random.randn(1)
U = np.concatenate((Ui,Uf,Up,Uo))


bi =np.random.randn(1)
bf =np.random.randn(1)
bp =np.random.randn(1)
bo =np.random.randn(1)
b = np.concatenate((bi,bf,bp,bo))



T = 3
prec_h = 10
prec_c = 8
E_z = np.random.uniform(0,1,size = (4,T)) #[Ezi;Ezf;Ezp;Ezo]
E_zozo = np.random.uniform(0,1,size = T)
E_v =  np.random.uniform(0,1,size=T)
E_omega = np.random.uniform(-1,1,size = (4,T)) #[Ewi;Ewf;Ewp;Ewo] 
E_gamma =  np.random.uniform(-1,1,size=T)
        
h_prec_hat, h_prec_mu_hat = update_qh(T, prec_h, E_z, E_v, E_omega, W)
Eh, Ehh = update_Eh(T, h_prec_hat, h_prec_mu_hat)

Ec, Ecc = update_qc(T, prec_c, E_gamma, E_z, E_v)

Ev, Evv = update_qv(T, prec_h, Ec, E_z, Eh, E_zozo)
print(Ev)
print(Evv)
