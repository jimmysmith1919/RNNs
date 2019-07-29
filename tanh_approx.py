import numpy as np
import matplotlib.pyplot as plt 




def tanh_approx(alpha, x):
    if x < -alpha:
        return -1
    elif x > alpha:
        return 1
    else:
        return -1+(x+alpha)/alpha

x = np.linspace(-2,2)
alpha = 3
sigma = 1
noise = np.random.normal(0,sigma, size=len(x))


y1 = np.tanh(x)+noise
y2 = np.zeros(len(x))
for i in range(0,len(x)):
    y2[i] = tanh_approx(alpha,x[i])
y2+= noise

plt.plot(x, y1, label='tanh')
plt.plot(x,y2, label = 'tanh_approx')
plt.legend()
plt.show()
