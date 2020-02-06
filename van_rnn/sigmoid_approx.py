import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit



alpha = 2.3

def sig_apprx(x,alpha):
    if x < -alpha:
        return 0
    elif x > alpha:
         return 1
    else:
        return 1/(2*alpha)*x+1/2






x = np.linspace(-10,10, num=1000)
#y = expit(x)
y = 2*expit(2*x)-1
plt.plot(x,y)


y2 = np.zeros(len(x))
for i in range(0,len(x)):
    y2[i] = 2*sig_apprx(2*x[i], alpha)-1
    #y2[i] = sig_apprx(x[i], alpha)

plt.plot(x, y2)


psi = np.log((1/3)/(1-1/3))
print(psi)
print(expit(psi))

plt.show()
