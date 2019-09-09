from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from  tensorflow.keras.optimizers import Adam
import time
import os

import matplotlib.pyplot as plt
import numpy as np

seed = 10
np.random.seed(seed)

d = 20
lr = .0001
eps = 100






model = Sequential()
model.add(LSTM(d, recurrent_activation='sigmoid', batch_input_shape=(1,1,1), return_sequences=True, 
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

Wi = weights[1][:,:d]
Wf = weights[1][:,d:2*d]
Wp = weights[1][:,2*d:3*d]
Wo = weights[1][:,3*d:4*d]

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
model.add(LSTM(d,recurrent_activation='sigmoid', batch_input_shape=(1,1,1), return_sequences=True, 
               stateful = True))
model.add(Dense(1) )
model.compile(optimizer=Adam(lr=lr), loss='mse')

model.set_weights(weights)




trainPred = model.predict(u)

plt.plot(t[1:], y_full.reshape(T_full))
plt.plot(t[1:stop], trainPred.reshape(T))

testPred = []
pred = trainPred[-1].reshape(1,1,1)




for j in range(0,T_new):
    pred= model.predict(pred)
    testPred.append(pred.reshape(1))
    pred = pred.reshape(1,1,1)                    


r = np.arange(stop,stop+T_new, dt)
plt.plot(r, testPred, 'r')






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
h_vec = []
c_vec = []

y_vec = []
for j in range(0,len(trainPred)):
    _, h, c = model2.predict(u[j,:,:].reshape(1,1,1))
    h_vec.append(h)
    c_vec.append(c)
    y = model3.predict(h.reshape(1,1,d))
    y_vec.append(y.reshape(1))

print('u')
print(u[:10])
print('h')
print(h_vec[:10])
print('c')
print(c_vec[:10])
print('y')
print(y_vec[:10])

plt.plot(t[1:stop], y_vec, 'm')




#One step at a time generateing new data
testPred = []
pred = y_vec[-1].reshape(1,1,1)

for j in range(0,T_new):
    _, h, c = model2.predict(pred)
    pred = model3.predict(h.reshape(1,1,d))
    testPred.append(pred.reshape(1))
    pred = pred.reshape(1,1,1)                    


r = np.arange(stop,stop+T_new, dt)
plt.plot(r, testPred, 'g')



plt.savefig(path+'/d{}_{}_epchs{}.png'.format(d, timestamp,eps))
plt.show()

