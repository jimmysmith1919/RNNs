from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from  tensorflow.keras.optimizers import Adam
from scipy.special import expit
import generate_GRU as gen
import time
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

seed = 10
np.random.seed(seed)

d = 10
lr = .0001
eps = 150

np.set_printoptions(precision=13)




model = Sequential()
model.add(GRU(d, recurrent_activation='sigmoid', 
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
T_new = 200 #Number of new steps to predict                         

t = np.arange(0, end, dt)
#data = np.sin((.06+.006)*t)
#data = np.sin((.05)*t)
data = np.sin((.2)*t)


ud = 1 #u dimension                                                            
yd = 1 #y dimension                                                            


T_full=len(data)-1


y_full = data[1:].reshape(T_full,ud,1)

stop = int(.8*end/dt)
T = stop-1
u = data[:T].reshape(T,ud,1)
y = data[1:stop].reshape(T)


T_test = T_full-T
u_test = data[T:-1].reshape(T_test,ud,1)
y_test = data[stop:].reshape(T_test)


#plt.plot(t[1:], y_full.reshape(T_full), label = 'true' )
#plt.show()
#sys.exit()

#eps = 100
#model.fit(u,y, epochs = eps, batch_size = 1, shuffle = False, verbose=2)
timestamp = time.time()
path = 'images/LSTM_{}'.format(timestamp)
os.mkdir(path)




for e in range(0,eps):
    model.fit(u,y, epochs = 1, batch_size = 1, shuffle = False, verbose=2)
    
    '''
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
    '''


    
trainPred = model.predict(u)
testPred = []
pred = trainPred[-1].reshape(1,1,1)
for j in range(0,T_new):
    pred= model.predict(pred)
    testPred.append(pred.reshape(1))
    pred = pred.reshape(1,1,1)                    

r = np.arange(stop,stop+T_new, dt)
#plt.plot(t[1:stop], trainPred.reshape(T), label='org_train1')
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

#Ui = weights[0][0,:d]
#Ur = weights[0][0,d:2*d]
#Up = weights[0][0,2*d:3*d]

Ui = weights[0][0,:d]
Ur = weights[0][0,d:2*d]
Up = weights[0][0,2*d:3*d]


Wi = weights[1][:,:d].T
Wr = weights[1][:,d:2*d].T
Wp = weights[1][:,2*d:3*d].T


bi = weights[2][:d]
br = weights[2][d:2*d]
bp = weights[2][2*d:3*d]

Wy = weights[3].reshape(1,d)
by = weights[4].reshape(1,1)

np.save('weights/GRU_d{}_eps{}_lr{}_end{}_{}'.format(d,eps, 
                                                      lr, end, 
                                                      timestamp),weights)

#create new model to reset h and c states
model = Sequential()
model.add(GRU(d,recurrent_activation='sigmoid', batch_input_shape=(1,1,1), return_sequences=True, stateful = True))
model.add(Dense(1) )
model.compile(optimizer=Adam(lr=lr), loss='mse')

model.set_weights(weights)




trainPred = model.predict(u)

plt.plot(t[1:], y_full.reshape(T_full), label = 'true' )
plt.plot(t[1:stop], trainPred.reshape(T), label='org_train2')

testPred = []
pred = trainPred[-1].reshape(1,1,1)




for j in range(0,T_new):
    pred= model.predict(pred)
    testPred.append(pred.reshape(1))
    pred = pred.reshape(1,1,1)                    


r = np.arange(stop,stop+T_new, dt)
plt.plot(r, testPred, label='org_test')






###Make new model to return cell and hidden states

inputs1 = Input(shape = (1,1), batch_size=1)
#inputs1 = Input(batch_shape=(1,1,1))
gru1, state_h = GRU(d, recurrent_activation = 'sigmoid',  
                               return_sequences=True, 
                               stateful = True, return_state=True)(inputs1)
model2 = Model(inputs=inputs1, outputs=[gru1, state_h])

    

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


y_vec = []
input = np.zeros((1,1,1))
print('train_results')
for j in range(0,len(trainPred)):
    #if j== 2:
    #    break
    _, h = model2.predict(u[j,:,:].reshape(1,1,1))
    #print(input)
    h_vec[j,:] = h
    #h_vec.append(h)
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

#check GRU


Ui = Ui.reshape(d,ud)
Ur = Ur.reshape(d,ud)
Up = Up.reshape(d,ud)

bi = bi.reshape(d,1)
br = br.reshape(d,1)
bp = bp.reshape(d,1)

#i_in = Wi @ h_vec[0].reshape(d,1) + Ui @ u[1,:,:].reshape(1,1)+bi
#r_in = Wr @ h_vec[0].reshape(d,1) + Ur @ u[1,:,:].reshape(1,1)+br
#p_in = Wp @ h_vec[0].reshape(d,1) + Up @ u[1,:,:].reshape(1,1)+bp



print(' ')
#Training data
h_vec = np.zeros((len(trainPred),d))
i_vec = np.zeros((len(trainPred),d))
r_vec = np.zeros((len(trainPred),d))
p_vec = np.zeros((len(trainPred),d))

h = np.zeros((d,1))
y_vec = []
input = np.zeros((1,1))

for j in range(0,len(trainPred)):
    i,r,p,h,y = gen.GRU_step(h,u[j,:,:].reshape(1,1), Wi, Ui, bi, Wr, Ur, br, Wp,
                             Up, bp, Wy, by)

    #y,c,h,i,_,_,_ = LSTM(c, h, u[j,:,:].reshape(1,1), Wi, Wf, Wp, Wo, 
    #                     Ui, Uf, Up, Uo, bi, bf, bp, bo, Wy, by)
    h_vec[j,:] = h.reshape(d)
    i_vec[j,:] = i.reshape(d)
    r_vec[j,:] = r.reshape(d)
    p_vec[j,:] = p.reshape(d)
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
plt.plot(t[1:stop], y_vec, label='My_GRU_train')

h_vec = np.zeros((T_new,d))
i_vec = np.zeros((T_new,d))
r_vec = np.zeros((T_new,d))
p_vec = np.zeros((T_new,d))

testPred = []
pred = y_vec[-1].reshape(1,1)
for j in range(0,T_new):
    i,r,p,h,pred  = gen.GRU_step(h, pred, Wi, Ui, bi, Wr, Ur, br, Wp,
                             Up, bp, Wy, by)
    
    #pred,c,h,i,f,p,o = LSTM(c, h, pred, Wi, Wf, Wp, Wo, 
    #                     Ui, Uf, Up, Uo, bi, bf, bp, bo, Wy, by)
    testPred.append(pred.reshape(1)) 

    h_vec[j,:] = h.reshape(d)
    
    i_vec[j,:] = i.reshape(d)
    r_vec[j,:] = r.reshape(d)
    p_vec[j,:] = p.reshape(d)


                                  
ran = np.arange(stop,stop+T_new, dt)                           
'''
for j in range(0,d):
    #plt.savefig(path+'/c_compare_{}.png'.format(j))
    plt.plot(ran,i_vec[:,j], label = 'i')
    plt.plot(ran,r_vec[:,j], label = 'r')
    plt.plot(ran,p_vec[:,j], label = 'p')
    plt.legend()
    plt.show()
    plt.close()
'''

plt.plot(ran, testPred, 'g', label='MY_GRU_test')

plt.legend()
plt.savefig(path+'/d{}_{}_epchs{}.png'.format(d, timestamp,eps))
plt.show()


