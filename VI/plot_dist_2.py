import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import expit
from matplotlib.patches import Ellipse
import sys



vars = np.zeros((48,5))
list = []
bin = [0,1]
v_bin = [0,1,2]
for i in bin:
    for f in bin:
        for p in bin:
            for o in bin:
                for v in v_bin:
                    list.append(np.array([i,f,p,o,v]))
print(list)
for i in range(0, len(list)):
    vars[i,:] = list[i] 


print(vars)

vars_h = vars[:6,3:]
print(vars_h)


vars_c = np.zeros((8,3))
for i in range(0,48,6):
    vars_c[int(i/6),:] = vars[i,:3]



print(vars_c)



def scale_calc(p):
    return -2*np.log(1-p)

def rad(sigma, s):
    return sigma*np.sqrt(s)

def bern_prob(p,x):
    return (p**x)*(1-p)**(1-x)


def LSTM(c0, h0, u, Wi, Wf, Wp, Wo, Ui, Uf, Up, Uo, bi, bf, bp, bo, Wy, by):
    i = expit(Wi @ h0 + Ui @ u + bi)                                         
    f = expit(Wf @ h0 + Uf @ u + bf)
    fp = Wp @ h0 + Up @ u + bp
    p = np.tanh(fp)#Wp @ h0 + Up @ u + bp)
    o = expit(Wo @ h0 + Uo @ u + bo)

    c = f*c0 + i*p
    h = o*np.tanh(c)
    y = Wy @ h + by
    return y,c,h,i,f,p,o, fp


d=20
ud = 1
yd = 1

end = 200
dt = 1
t = np.arange(0, end, dt)
data = np.sin((.06+.006)*t)


'''
Wi = np.random.uniform(-1,1, size=(d,d))                                     
Wf = np.random.uniform(-1,1, size=(d,d))                                     
Wp = np.random.uniform(-1,1, size=(d,d))                                     
Wo = np.random.uniform(-1,1, size=(d,d))                                     
Wy = np.random.uniform(-1,1, size=(yd,d))                                    
                                                                             
#Wy = np.zeros((1,d))                                                        
#Wy[0,0] = 1                                                                 
                                                                             
                                                                             
Ui = np.random.uniform(-1,1, size=(d,ud))                                    
Uf = np.random.uniform(-1,1, size=(d,ud))                                    
Up = np.random.uniform(-1,1, size=(d,ud))                                    
Uo = np.random.uniform(-1,1, size=(d,ud))                                    
                                                                             
#Ui = np.zeros((d,1))                                                        
#Uf = np.zeros((d,1))                                                        
#Up = np.zeros((d,1))                                                        
#Uo = np.zeros((d,1))                                                        
                                                                             
                                                                             
bi = np.random.uniform(-1,1, size=(d,1))                                     
bf = np.random.uniform(-1,1, size=(d,1))                                     
bp = np.random.uniform(-1,1, size=(d,1))                                     
bo = np.random.uniform(-1,1, size=(d,1))                                     
by = np.random.uniform(-1,1, size=(yd,1))                                    
#by = np.zeros((yd,1))                                                       
Uy = np.zeros((yd,ud)) 
'''



#Loaded weights                                                              
wfile = 'weights/'+'LSTM_d20_eps100_lr0.0001_end200_1569288491.5787299.npy'
weights = np.load(wfile)

Ui = weights[0][0,:d].reshape(d,ud)
Uf = weights[0][0,d:2*d].reshape(d,ud)
Up = weights[0][0,2*d:3*d].reshape(d,ud)
Uo = weights[0][0,3*d:4*d].reshape(d,ud)

Wi = weights[1][:,:d].T
Wf = weights[1][:,d:2*d].T                                                
Wp = weights[1][:,2*d:3*d].T
Wo = weights[1][:,3*d:4*d].T

bi = weights[2][:d].reshape(d,1)
bf = weights[2][d:2*d].reshape(d,1)
bp = weights[2][2*d:3*d].reshape(d,1)
bo = weights[2][3*d:4*d].reshape(d,1)

Wy = weights[3].reshape(yd,d) 
by = weights[4].reshape(yd,1)


var_c = .01
var_h = .01





##LSTM draw
steps = 115
alpha= 1.5
c = np.zeros((d,1))
h = np.zeros((d,1))
for j in range(0,steps):
    u = data[j].reshape(ud,1)
    y,c,h,i,f,p,o,fp = LSTM(c, h, u, 
                     Wi, Wf, Wp, Wo, 
                     Ui, Uf, Up, Uo, 
                     bi, bf, bp, bo, Wy, by)
    if j == steps-2:
        c_minus = c[0]
'''
##Plot mixtures
s1 = scale_calc(.68)
rad1_c = rad(np.sqrt(var_c), s1)
rad1_h = rad(np.sqrt(var_h), s1)

s2 = scale_calc(.95)
rad2_c = rad(np.sqrt(var_c), s2)
rad2_h = rad(np.sqrt(var_h), s2)

s3 = scale_calc(.997)
rad3_c = rad(np.sqrt(var_c), s3)
rad3_h = rad(np.sqrt(var_h), s3)

fig, ax = plt.subplots(1,1,subplot_kw={'aspect':'equal'})
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)


cmap = plt.cm.get_cmap('hsv',32)
'''


for j in range(0,len(list)):
    var = vars[j,:]
    zi = var[0]
    zf = var[1]
    zp = var[2]
    zo = var[3]
    v = var[4]


    int = np.random.randint(0,32)
    color = cmap(int)
                

    mean_c = zf*c_minus+zi*(2*zp-1) 
    
    if v == 0:
        vc = -1
    elif v ==1:
        vc = -1+1/alpha*(mean_c+alpha)
    else:
        vc = 1
    
    
    mean_h = zo*vc
    print('next')
    print('mean_c')
    print(mean_c)
    print('zo')
    print(zo)
    print('v')
    print(v)
    print('vc')
    print(vc)
    print('mean_h')
    print(mean_h)
    
    ellipse1 = Ellipse((mean_c,mean_h),2*rad1_c, 2*rad1_h , facecolor='none', 
                       edgecolor=color)

    ellipse2 = Ellipse((mean_c,mean_h),2*rad2_c, 2*rad2_h , facecolor='none',
                       edgecolor=color)

    ellipse3 = Ellipse((mean_c,mean_h),2*rad3_c, 2*rad3_h , facecolor='none',
                       edgecolor=color)


    ax.add_patch(ellipse1)
    ax.add_patch(ellipse2)
    ax.add_patch(ellipse3)


plt.xlabel('c: mix_mean=zf*c_{t-1}+zi(2zp-1)')
plt.ylabel('h: mix_mean= zo(2v-1)')
plt.plot(c[0,:], h[0,:], 'ro', label = 'LSTM')

plt.show()

sys.exit()






pi = i[0]
pf = f[0]
pp = expit(2*fp[0])
po = o[0]


sum_c = 0
sum_pc = 0
sum_ph = 0
sum_h = 0


for j in range(0,len(vars_c[:,0])):
    var = vars_c[j,:]
    zi = var[0]
    zf = var[1]
    zp = var[2]

    prob_i = bern_prob(pi,zi)
    prob_f = bern_prob(pf,zf)
    prob_p = bern_prob(pp,zp)
    
    sum_pc +=prob_i*prob_f*prob_p

    mean_c = zf*c_minus+zi*(2*zp-1) 
    
    sum_c += mean_c*prob_i*prob_f*prob_p






pv1 = expit(2*sum_c)

for j in range(0,len(vars_h[:,0])):
    var = vars_h[j,:]
    zo = var[0]
    v = var[1]
    
    prob_o = bern_prob(po,zo)
    prob_v = bern_prob(pv1,v) 

    sum_ph += prob_o*prob_v
    
    mean_h = zo*(2*v-1)
    sum_h += mean_h*prob_o*prob_v




print('c')
print(sum_pc)
print('{0:.16f}'.format(sum_c[0]))
print('{0:.16f}'.format(c[0,:][0]))

print('h')
print(sum_ph)
print('{0:.16f}'.format(sum_h[0]))
print('{0:.16f}'.format(h[0,:][0]))



plt.plot(c[0,:], h[0,:], 'ro', label = 'LSTM')
plt.plot(sum_c, sum_h, 'bx', label='mixture_mean')



N = 10000
sum_h = 0
sum_c = 0
zi_sum = 0
zf_sum = 0
zp_sum = 0

for j in range(0,N):
    zi = np.random.binomial(1, pi)
    zi_sum += zi
    zf = np.random.binomial(1, pf)
    zf_sum += zf
    zp = np.random.binomial(1, pp)
    zp_sum += zp
    cm = np.random.normal(zf*c_minus+zi*(2*zp-1), np.sqrt(var_c))
    v = np.random.binomial(1, expit(2*cm))
    zo = np.random.binomial(1, po)
    hm = np.random.normal(zo*(2*v-1), np.sqrt(var_h))
    
    sum_c += cm
    sum_h += hm

c = sum_c/N
h = sum_h/N

print('sample')
print('c')
print(c)
print('h')
print(h)

plt.plot(c, h, 'm*', label='Samples={}'.format(N))

c_avg = zf_sum/N*c_minus+zi_sum/N*(2*zp_sum/N-1)

v_sum = 0
zo_sum = 0
for j in range(0,N):
    v = np.random.binomial(1, expit(2*c_avg))
    zo = np.random.binomial(1, po)
    v_sum += v
    zo_sum += zo

h_avg = hm = zo_sum/N*(2*v_sum/N-1)

plt.plot(c_avg, h_avg, 'g.', label='split_Samples={}'.format(N))

plt.title('T={}'.format(steps-1))
plt.legend()
plt.show()
