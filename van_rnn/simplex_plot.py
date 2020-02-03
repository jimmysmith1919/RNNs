import numpy as np
import matplotlib.pyplot as plt
import ternary
from scipy.special import expit

## Boundary and Gridlines
scale = 1
figure, tax = ternary.figure(scale=scale)

# Draw Boundary and Gridlines
tax.boundary(linewidth=2.0)
tax.gridlines(color="black", multiple=.1)
tax.gridlines(color="blue", multiple=1, linewidth=0.5)



# Set ticks
tax.ticks(axis='lbr', linewidth=1)

# Remove default Matplotlib Axes
tax.clear_matplotlib_ticks()





alpha = 2.5
tau_list = [.1, 1,2, 5, 10, 100]
ran = 30
x = np.linspace(-ran,ran, num=1000)



def fun_x(x, alpha, tau):
    cat1 = expit((-x-alpha)/tau)
    cat2 = (1-cat1)*expit((x-alpha)/tau)
    cat3 = 1-cat1-cat2

    cat1 = cat1.reshape(-1,1)
    cat2 = cat2.reshape(-1,1)
    cat3 = cat3.reshape(-1,1)

    return np.concatenate((cat1,cat2,cat3), axis=1)
    

def get_points(x, fx):
    points = []
    for i in range(0,len(x)):
        points.append(fx[i,:])
    return points

'''
##Different taus, all x
tau = .0001
fx = fun_x(x, alpha, tau)
points = get_points(x,fx)

tax.plot(points,marker='D', label = 'tau={}'.format(tau))


for tau in tau_list:
    fx = fun_x(x, alpha, tau)
    points = get_points(x,fx)
    tax.plot(points, label = 'tau={}'.format(tau))



tau = 10000
fx = fun_x(x, alpha, tau)
points = get_points(x,fx)

tax.plot(points,marker='D', label = 'tau={}'.format(tau))
tax.plot([(1/3,1/3,1/3)],marker='D', label='Center')
'''

'''
tau_list = [.0001, .1, 1, 2,5, 10, 100 ]

check = 10
point = check*np.ones(1)

for tau in tau_list:
    fx = fun_x(point, alpha, tau)
    points = get_points(point, fx)
    tax.plot(points,marker='D', label = 'x={},tau={}'.format(check,tau))
'''

tau = 3

check = -2.6
point = check*np.ones(1)
fx = fun_x(point, alpha, tau)
points = get_points(point, fx)
tax.plot(points,marker='D', label = 'x={},tau={}'.format(check,tau))

check = 2.6
point = check*np.ones(1)
fx = fun_x(point, alpha, tau)
points = get_points(point, fx)
tax.plot(points,marker='D', label = 'x={},tau={}'.format(check,tau))

check = 0
point = check*np.ones(1)
fx = fun_x(point, alpha, tau)
points = get_points(point, fx)
tax.plot(points,marker='D', label = 'x={},tau={}'.format(check,tau))







tax.plot([(1/3,1/3,1/3)],marker='D', label='Center')

# Set Axis labels and Title
fontsize = 10
tax.set_title("Simplex, x_range=-{}-{}, alpha={}".format(ran,ran, alpha), fontsize=fontsize)
tax.left_axis_label("Cat f(x)", fontsize=fontsize)
tax.right_axis_label("Cat 1", fontsize=fontsize)
tax.bottom_axis_label("Cat neg1", fontsize=fontsize)
tax.legend()    
ternary.plt.show()
