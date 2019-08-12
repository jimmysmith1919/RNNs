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

def 



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
prec_h = .3
E_z = np.random.uniform(0,1,size = (4,T)) #[Ezi;Ezf;Ezp;Ezo]
E_v =  np.random.uniform(0,1,size=T)
E_omega = np.random.uniform(-1,1,size = (4,T)) #[Ewi;Ewf;Ewp;Ewo] 

        
h_prec_hat, h_prec_mu_hat = update_qh(T, prec_h, E_z, E_v, E_omega, W)

print(h_prec_hat)
print(h_prec_mu_hat)



