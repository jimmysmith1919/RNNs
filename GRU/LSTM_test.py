from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from  tensorflow.keras.optimizers import Adam
from scipy.special import expit
import time
import os

import matplotlib.pyplot as plt
import numpy as np

seed = 10
np.random.seed(seed)

d = 20
lr = .0001
eps = 30






model = Sequential()
model.add(LSTM(d, recurrent_activation='sigmoid', 
               batch_input_shape=(1,1,1), return_sequences=True, 
               stateful = True))
model.add(Dense(1) )
model.compile(optimizer=Adam(lr=lr), loss='mse')


'''
inputs1 = Input(shape = (1,1), batch_size=1)
#inputs1 = Input(batch_shape=(1,1,1))
lstm1, state_h, state_c = LSTM(d,  
                               return_sequences=False, 
                               stateful = True, return_state=True)(inputs1)
dense = Dense(1)(lstm1)
model = Model(inputs=inputs1, outputs=[dense, state_h, state_c])
model.compile(optimizer=Adam(lr=lr), loss='mse')
#model.summary()
'''




end = 200
dt = 1
T_new = int(.2*end/dt)+200 #Number of new steps to predict                         

t = np.arange(0, end, dt)
data = np.sin((.06+.006)*t)

ud = 1 #u dimension                                                            
yd = 1 #y dimension                                                            

#T=len(data)-1                                                                 
T_full=len(data)-1


#u = data[:-1].reshape(T,ud,1)                                                 
#y = data[1:].reshape(T,yd,1)                                                  
u_full = data[:-1].reshape(T_full,ud,1)
#u_full = np.zeros((T_full,1)).reshape(T_full,ud,1)

y_full = data[1:].reshape(T_full,ud,1)

stop = int(.8*end/dt)#int(3*len(u_full)/4)+1                                  
T = stop-1
u = data[:stop-1].reshape(T,ud,1)
y = data[1:stop].reshape(T)#,yd,1)


T_test = T_full-T
u_test = data[stop-1:-1].reshape(T_test,ud,1)
y_test = data[stop:].reshape(T_test)

#eps = 100
#model.fit(u,y, epochs = eps, batch_size = 1, shuffle = False, verbose=2)
timestamp = time.time()
path = 'images/LSTM_{}'.format(timestamp)
os.mkdir(path)




for e in range(0,eps):
    model.fit(u,y, epochs = 1, batch_size = 1, shuffle = False, verbose=2)
    
    
    if e%10 == 0:
        trainPred = model.predict(u)
        testPred = []
        pred = trainPred[-1].reshape(1,1,1)
        for j in range(0,T_new):
            pred= model.predict(pred)
            testPred.append(pred.reshape(1))
            pred = pred.reshape(1,1,1)                    

        r = np.arange(stop,stop+T_new, dt)
        plt.plot(r, testPred, label = '{}'.format(e))
        
plt.legend()
plt.savefig(path+'/d{}_{}_epch{}.png'.format(d, timestamp,e))
plt.close()



    
    

trainPred = model.predict(u)
testPred = []
pred = trainPred[-1].reshape(1,1,1)
for j in range(0,T_new):
    pred= model.predict(pred)
    testPred.append(pred.reshape(1))
    pred = pred.reshape(1,1,1)                    

r = np.arange(stop,stop+T_new, dt)
plt.plot(r, testPred, label = '{}'.format(e))



weights = model.get_weights()
'''
print(len(weights))
print(weights[0].shape)
print(weights[0])
print(weights[1].shape)
print(weights[1])
print(weights[2].shape)
print(weights[2])
print(weights[3].shape)
print(weights[3])
print(weights[4].shape)
print(weights[4])
'''

Ui = weights[0][0,:d]
Uf = weights[0][0,d:2*d]
Up = weights[0][0,2*d:3*d]
Uo = weights[0][0,3*d:4*d]

Wi = weights[1][:,:d].T
Wf = weights[1][:,d:2*d].T
Wp = weights[1][:,2*d:3*d].T
Wo = weights[1][:,3*d:4*d].T

bi = weights[2][:d]
bf = weights[2][d:2*d]
bp = weights[2][2*d:3*d]
bo = weights[2][3*d:4*d]

Wy = weights[3].reshape(1,d)

by = weights[4].reshape(1,1)

np.save('weights/LSTM_d{}_eps{}_lr{}_end{}_{}'.format(d,eps, 
                                                      lr, end, 
                                                      timestamp),weights)

#create new model to reset h and c states
model = Sequential()
model.add(LSTM(d,recurrent_activation='sigmoid', batch_input_shape=(1,1,1), return_sequences=True, stateful = True))
model.add(Dense(1) )
model.compile(optimizer=Adam(lr=lr), loss='mse')

model.set_weights(weights)




trainPred = model.predict(u)

plt.plot(t[1:], y_full.reshape(T_full))
plt.plot(t[1:stop], trainPred.reshape(T), label = 'train_org')

testPred = []
pred = trainPred[-1].reshape(1,1,1)




for j in range(0,T_new):
    pred= model.predict(pred)
    testPred.append(pred.reshape(1))
    pred = pred.reshape(1,1,1)                    


r = np.arange(stop,stop+T_new, dt)
plt.plot(r, testPred, 'r', label='new1')






###Make new model to return cell and hidden states

inputs1 = Input(shape = (1,1), batch_size=1)
#inputs1 = Input(batch_shape=(1,1,1))
lstm1, state_h, state_c = LSTM(d, recurrent_activation = 'sigmoid',  
                               return_sequences=True, 
                               stateful = True, return_state=True)(inputs1)
model2 = Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])

    

#set with just LSTM weights
mod_weights = weights[:3]
model2.set_weights(mod_weights)



#New model that handles y output
inputs2 = Input(shape = (1,d))
dense = Dense(1)(inputs2)
model3 = Model(inputs=inputs2, outputs=dense)

mod_weights2 = weights[3:]
model3.set_weights(mod_weights2)


#One step at a time on training data
h_vec = np.zeros((len(trainPred),d))
c_vec = np.zeros((len(trainPred),d))

y_vec = []
input = np.zeros((1,1,1))
for j in range(0,len(trainPred)):
    #if j== 2:
    #    break
    _, h, c = model2.predict(u[j,:,:].reshape(1,1,1))
    #print(input)
    h_vec[j,:] = h
    c_vec[j,:] = c
    #h_vec.append(h)
    #c_vec.append(c)
    y = model3.predict(h.reshape(1,1,d))
    y_vec.append(y.reshape(1))

###Print Eh and h prior                                                     \
'''                                                                             
for j in range(0,d):
    plt.plot(t[1:stop],h_vec[:,j], label = 'Prior')
    #plt.savefig(path+'/h_compare_{}.png'.format(j))
    plt.show()
    plt.close()
'''
'''
###Print Ec and c prior                                                     \
plt.close()                
                                                             
for j in range(0,d):
    plt.plot(t[1:stop],c_vec[:,j], label = 'Prior')
    #plt.savefig(path+'/c_compare_{}.png'.format(j))
    plt.legend()
    plt.show()
    plt.close()
'''



'''
plt.plot(t[1:stop], y_vec, 'm')




#One step at a time generating new data
testPred = []
pred = y_vec[-1].reshape(1,1,1)

for j in range(0,T_new):
    _, h, c = model2.predict(pred)
    pred = model3.predict(h.reshape(1,1,d))
    testPred.append(pred.reshape(1))
    pred = pred.reshape(1,1,1)                    


r = np.arange(stop,stop+T_new, dt)
plt.plot(r, testPred, 'g', label='new2')
'''

#check LSTM
def LSTM(c0, h0, u, Wi, Wf, Wp, Wo, Ui, Uf, Up, Uo, bi, bf, bp, bo, Wy, by):
    i = expit(Wi @ h0 + Ui @ u + bi)
    #print('Wih')
    #print(Wi @ h0)
    f = expit(Wf @ h0 + Uf @ u + bf)
    p = np.tanh(Wp @ h0 + Up @ u + bp)
    o = expit(Wo @ h0 + Uo @ u + bo)

    c = f*c0 + i*p
    h = o*np.tanh(c)
    y = Wy @ h + by
    return y,c,h,i,f,p,o





Ui = Ui.reshape(d,ud)
Uf = Uf.reshape(d,ud)
Up = Up.reshape(d,ud)
Uo = Uo.reshape(d,ud)



bi = bi.reshape(d,1)
bf = bf.reshape(d,1)
bp = bp.reshape(d,1)
bo = bo.reshape(d,1)


i_in = Wi @ h_vec[0].reshape(d,1) + Ui @ u[1,:,:].reshape(1,1)+bi
f_in = Wf @ h_vec[0].reshape(d,1) + Uf @ u[1,:,:].reshape(1,1)+bf
p_in = Wp @ h_vec[0].reshape(d,1) + Up @ u[1,:,:].reshape(1,1)+bp
o_in = Wo @ h_vec[0].reshape(d,1) + Uo @ u[1,:,:].reshape(1,1)+bo


print(' ')
#Training data
h_vec = np.zeros((len(trainPred),d))
c_vec = np.zeros((len(trainPred),d))
h = np.zeros((d,1))
c = np.zeros((d,1))
y_vec = []
input = np.zeros((1,1))
for j in range(0,len(trainPred)):
    #y,c,h,i,_,_,_ = LSTM(c, h, input, Wi, Wf, Wp, Wo, 
    #                     Ui, Uf, Up, Uo, bi, bf, bp, bo, Wy, by)
    y,c,h,i,_,_,_ = LSTM(c, h, u[j,:,:].reshape(1,1), Wi, Wf, Wp, Wo, 
                         Ui, Uf, Up, Uo, bi, bf, bp, bo, Wy, by)
    h_vec[j,:] = h.reshape(d)
    c_vec[j,:] = c.reshape(d)
    #if j==2:
    #    break
    
    #print(input)
    y_vec.append(y.reshape(1))
'''
###Print Ec and c prior                                                     \
plt.close()                
                                                             
for j in range(0,d):
    plt.plot(t[1:stop],c_vec[:,j], label = 'Prior')
    #plt.savefig(path+'/c_compare_{}.png'.format(j))
    plt.legend()
    plt.show()
    plt.close()
'''
plt.plot(t[1:stop], y_vec, label='My_LSTM_train')
plt.legend()
plt.show()

h_vec = np.zeros((T_new,d))
c_vec = np.zeros((T_new,d))
i_vec = np.zeros((T_new,d))
f_vec = np.zeros((T_new,d))
p_vec = np.zeros((T_new,d))
o_vec = np.zeros((T_new,d))

testPred = []
pred = y_vec[-1].reshape(1,1)
for j in range(0,T_new):
    pred,c,h,i,f,p,o = LSTM(c, h, pred, Wi, Wf, Wp, Wo, 
                         Ui, Uf, Up, Uo, bi, bf, bp, bo, Wy, by)
    testPred.append(pred.reshape(1)) 

    h_vec[j,:] = h.reshape(d)
    c_vec[j,:] = c.reshape(d)
    
    i_vec[j,:] = i.reshape(d)
    f_vec[j,:] = f.reshape(d)
    p_vec[j,:] = p.reshape(d)
    o_vec[j,:] = o.reshape(d)


plt.close()                
                                  
r = np.arange(stop,stop+T_new, dt)                           
'''
for j in range(0,d):
    plt.plot(r,c_vec[:,j], label = 'c')
    #plt.savefig(path+'/c_compare_{}.png'.format(j))
    plt.plot(r,i_vec[:,j], label = 'i')
    plt.plot(r,f_vec[:,j], label = 'f')
    plt.plot(r,p_vec[:,j], label = 'p')
    plt.plot(r,o_vec[:,j], label = 'o')
    plt.legend()
    plt.show()
    plt.close()
'''

plt.plot(r, testPred, 'g', label='MY_LSTM_test')

plt.legend()
plt.savefig(path+'/d{}_{}_epchs{}.png'.format(d, timestamp,eps))
plt.show()

