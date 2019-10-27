import numpy as np
import matplotlib.pyplot as plt
import generate_GRU as gen

seed = np.random.randint(0,10000)

d = 10 #hidden state dimension
ud = 3 # input dimension
yd = ud

#Random weights                                                               
Wi = np.random.uniform(-1,1, size=(d,d))                                      
Wr = np.random.uniform(-1,1, size=(d,d))                                      
Wp = np.random.uniform(-1,1, size=(d,d))                                      
Wy = np.random.uniform(-1,1, size=(yd,d))                                     

Ui = np.random.uniform(-1,1, size=(d,ud))                                     
Ur = np.random.uniform(-1,1, size=(d,ud))                                     
Up = np.random.uniform(-1,1, size=(d,ud))                                     

bi = np.random.uniform(-1,1, size=d)                                      
br = np.random.uniform(-1,1, size=d)                                      
bp = np.random.uniform(-1,1, size=d)                                      
by = np.random.uniform(-1,1, size=yd) 

steps = 100
h_0 = 0*np.ones(d)
u_0 = .4*np.ones(ud)


i_vec,r_vec,p_vec,h_vec,y_vec = gen.generate_rec_inp(gen.GRU_step, steps, 
                                                         h_0, u_0,
                                                         Wi, Ui, bi,
                                                         Wr, Ur, br,
                                                         Wp, Up, bp,
                                                         Wy, by)




i_vec,r_vec,p_vec,h_vec,y_vec2 = gen.generate_rec_inp(gen.stoch_GRU_step_mean,
                                                     steps, 
                                                     h_0, u_0,
                                                     Wi, Ui, bi,
                                                     Wr, Ur, br,
                                                     Wp, Up, bp,
                                                     Wy, by)

