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
        prec_hat[t] = 4*Ezozo[t,t]*prec_h
        prec_mu_hat[t] = 2*Ec[t]+2*Ezozo[t,t]*prec_h+2*Ez[3,t]*Eh[t]*prec_h
    
    covar = np.linalg.inv(np.diag(prec_hat))
    Ev = covar @ prec_mu_hat
    Evv = np.outer(Ev, Ev)+covar
    return Ev, Evv
    


def update_omega_star(T, Eh, Ehh, u, W_star, U_star, b_star):
    b=np.ones(T)
    Eh = np.insert(Eh, 0, 0)
    Eh = np.delete(Eh, -1)
    Ehh = np.insert(Ehh, 0, 0)
    Ehh = np.delete(Ehh, -1)
    c1 = Ehh*W_star**2
    print('c1:', c1)
    c2 = ( 2*U_star*u*W_star+2*b_star*W_star )*Eh
    print('c2:', c2)
    c3 = ( U_star*u )**2 + 2*U_star*u*b_star + np.ones(T)*b_star**2
    print('c3:', c3)
    c = np.sqrt(c1+c2+c3)
    return  b/(2*c)*np.tanh(c/2)

def update_gamma(T, Ecc):
    b = np.ones(T)
    Ecc = np.diagonal(Ecc)
    c = 2*np.sqrt(Ecc)
    return  b/(2*c)*np.tanh(c/2)

    
def update_qzi(T, prec_c, Eh, Ec, Ezp, Ezpzp, Ezf, Wi, Ui, bi, u):
    prec_hat = np.zeros(T)
    prec_mu_hat = np.zeros(T)
    Eh = np.insert(Eh, 0, 0) #now Eh[t] = Eh[t-1]
    Ec = np.insert(Ec, 0, 0) #same ^

    for t in range(0,T):
        prec_hat[t] = 4*Ezpzp[t,t]*prec_c+(-4*Ezp[t]*prec_c)+prec_c
        prec_mu_hat[t] = Wi*Eh[t]+Ui*u[t]+bi
        prec_mu_hat[t] += -Ezf[t]*Ec[t]*(2*Ezp[t]-1)*prec_c
        prec_mu_hat[t] += (2*Ezp[t]-1)*(Ec[t+1]*prec_c)
        
    
    covar = np.linalg.inv(np.diag(prec_hat))
    Ezi = covar @ prec_mu_hat
    Ezizi = np.outer(Ezi, Ezi)+covar
    return Ezi, Ezizi

    
def update_qzf(T, prec_c, Eh, Ec, Ecc, Ezp, Ezi, Wf, Uf, bf, u):
    prec_hat = np.zeros(T)
    prec_mu_hat = np.zeros(T)
    #Eh = np.insert(Eh, 0, 0) #now Eh[t] = Eh[t-1]
    #Ecc = np.concatenate( (np.zeros((1,T)), Ecc), axis= 0)
    #Ecc = np.concatenate((np.zeros((T+1,1)), Ecc), axis= 1)

    for t in range(1,T):
        prec_hat[t] = 4*Ecc[t-1,t-1]*prec_c
        prec_mu_hat[t] = Wf*Eh[t-1]+Uf*u[t]+bf
        prec_mu_hat[t] += Ecc[t-1,t]*prec_c
        prec_mu_hat[t] += -Ec[t-1]*( (Ezi[t]*(2*Ezp[t]-1))*prec_c)
        
    
    covar = np.linalg.inv(np.diag(prec_hat[1:]))
    Ezf = covar @ prec_mu_hat[1:]
    Ezfzf = np.outer(Ezf, Ezf)+covar
    
    #hard coding Ez1 as zero, need to check this makes sense
    Ezf = np.insert(Ezf, 0, 0)
    Ezfzf = np.concatenate((np.zeros((1,T-1)), Ezfzf), axis = 0)
    Ezfzf = np.concatenate( (np.zeros((T,1)), Ezfzf), axis =1)
    return Ezf, Ezfzf

    
def update_qzp(T, prec_c, Eh, Ec, Ezi, Ezizi, Ezf, Wp, Up, bp, u):
    prec_hat = np.zeros(T)
    prec_mu_hat = np.zeros(T)
    Eh = np.insert(Eh, 0, 0) #now Eh[t] = Eh[t-1]
    Ec = np.insert(Ec, 0, 0) #same ^

    for t in range(0,T):
        prec_hat[t] = 4*Ezizi[t,t]*prec_c
        prec_mu_hat[t] = Wp*Eh[t]+Up*u[t]+bp
        prec_mu_hat[t] += 2*Ezizi[t,t]*prec_c
        prec_mu_hat[t] += -2*Ezi[t]*Ezf[t]*Ec[t]*prec_c
        prec_mu_hat[t] += 2*Ezi[t]*Ec[t+1]*prec_c
        
    
    covar = np.linalg.inv(np.diag(prec_hat))
    Ezp = covar @ prec_mu_hat
    Ezpzp = np.outer(Ezp, Ezp)+covar
    return Ezp, Ezpzp

def update_qzo(T, prec_h, Eh, Ev, Evv, Wo, Uo, bo, u):
    prec_hat = np.zeros(T)
    prec_mu_hat = np.zeros(T)
    Eh = np.insert(Eh, 0, 0) #now Eh[t] = Eh[t-1]

    for t in range(0,T):
        prec_hat[t] = 4*Evv[t,t]*prec_h+-4*Ev[t]*prec_h+prec_h
        prec_mu_hat[t] = Wo*Eh[t]+Uo*u[t]+bo
        prec_mu_hat[t] += (2*Ev[t]-1)*Eh[t+1]*prec_h
        
    
    covar = np.linalg.inv(np.diag(prec_hat))
    Ezo = covar @ prec_mu_hat
    Ezozo = np.outer(Ezo, Ezo)+covar
    return Ezo, Ezozo


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



T = 10
u = np.ones(T)
prec_h = 1
prec_c = 1
Ez = 0.5*np.ones((4,T)) #[Ezi;Ezf;Ezp;Ezo]
Ezozo = np.outer(Ez[3,:], Ez[3,:])
Ezpzp = np.outer(Ez[2,:], Ez[2,:])
Ev =  .5*np.ones(T)
E_omega = np.zeros((4,T)) #[Ewi;Ewf;Ewp;Ewo] 
E_gamma =  np.zeros(T)

print(E_omega.shape)

for k in range(0,20):
    Ec, Ecc = update_qc(T, prec_c, E_gamma, Ez, Ev)

    prec_h_hat, prec_mu_h_hat = update_qh(T, prec_h, Ez, Ev, E_omega, W)
    Eh, Ehh = update_Eh(T, prec_h_hat, prec_mu_h_hat)

    Ev, Evv = update_qv(T, prec_h, Ec, Ez, Eh, Ezozo)

    E_omega = np.zeros((4,T))
    E_omega[0,:] = update_omega_star(T, Eh, Ehh, u, Wi, Ui, bi)
    E_omega[1,:] = update_omega_star(T, Eh, Ehh, u, Wf, Uf, bf)
    E_omega[2,:] = update_omega_star(T, Eh, Ehh, u, Wp, Up, bp)
    E_omega[3,:] = update_omega_star(T, Eh, Ehh, u, Wo, Uo, bo)

    E_gamma = update_gamma(T, Ecc)


    Ez = np.zeros((4,T))
    Ez[0,:], Ezizi = update_qzi(T, prec_c, Eh, Ec, Ez[2,:], Ezpzp,  Ez[1,:],
                                 Wi, Ui, bi, u)
    Ez[1,:], Ezfzf = update_qzf(T, prec_c, Eh, Ec, Ecc, Ez[2,:], Ez[0,:], 
                                 Wf, Uf, bf, u)

    Ez[2,:], Ezpzp = update_qzp(T, prec_c, Eh, Ec, Ez[0,:], Ezizi, Ez[1,:], 
                                Wp, Up, bp, u)

    Ez[3,:], Ezozo = update_qzo(T, prec_h, Eh, Ev, Evv, Wo, Uo, bo, u)
    
    



