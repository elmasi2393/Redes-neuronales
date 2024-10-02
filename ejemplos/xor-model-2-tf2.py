"""
File: xor-model.py
Author: German Mato
Email: matog@cab.cnea.gov.ar
Description: Codificacion del XOR modelo 2 - version tensorflow 2 - modified accuracy
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

seed=1                            # for reproducibility 
np.random.seed(seed)
tf.random.set_seed(seed)


def output_activation(x):
    return  tf.math.sinh(x)/tf.math.cosh(x)

# Network architecture
hidden_dim=1 # Number of hidden units

inputs = tf.keras.layers.Input(shape=(2,))
x = tf.keras.layers.Dense(hidden_dim, activation=output_activation)(inputs)
merge=tf.keras.layers.concatenate([inputs,x],axis=-1)
predictions = tf.keras.layers.Dense(1, activation=output_activation)(merge)


# Data Input

ntrain=4
x_train=np.zeros((ntrain,2), dtype=np.float32)
y_train=np.zeros((ntrain,1), dtype=np.float32)
ntest=4
x_test=np.zeros((ntest,2), dtype=np.float32)
y_test=np.zeros((ntest,1), dtype=np.float32)

x_train[0,0]=1
x_train[0,1]=1
y_train[0]=1

x_train[1,0]=-1
x_train[1,1]=1
y_train[1]=-1

x_train[2,0]=1
x_train[2,1]=-1
y_train[2]=-1

x_train[3,0]=-1
x_train[3,1]=-1
y_train[3]=1

print(x_train.shape)
print(y_train.shape)

x_test[:]=x_train
y_test[:]=y_train
print(x_test.shape)
print(y_test.shape)

# accuracy compatible with tensorflow v1
# from tensorflow.python.keras import backend as K
# def v1_accuracy(y_true, y_pred):
#      return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)
 
# # Model 
# opti=tf.keras.optimizers.Adam(lr=0.01, decay=0.0)
# model = tf.keras.Model(inputs=inputs, outputs=predictions)
# model.compile(optimizer=opti,
#               loss='MSE',metrics=[v1_accuracy])

# Crear el modelo
model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              loss='mse',  # Error cuadrático medio
              metrics=['accuracy'])



history=model.fit(x=x_train, y=y_train,
                epochs=500,
                batch_size=4,
                shuffle=False,
                validation_data=(x_test, y_test), verbose=True)
#

tf.keras.utils.plot_model(model, to_file='model-2.png', show_shapes=False, show_layer_names=True, rankdir='TB')
print(model.summary())
encoded_log = model.predict(x_test, verbose=True)
print(encoded_log.shape)


#####################################################################
# Output files
fout=open("xor-out-2.dat","wb")
ftrain=open("xor-train-2.dat","wb")
ftest=open("xor-test-2.dat","wb")
#
np.savetxt(ftrain,np.c_[x_train,y_train],delimiter=" ")
np.savetxt(ftest,np.c_[x_test, y_test],delimiter=" ")
np.savetxt(fout,np.c_[x_test, encoded_log],delimiter=" ")

W_Input_Hidden = model.layers[1].get_weights()[0]
W_Output_Hidden = model.layers[3].get_weights()[0]
B_Input_Hidden = model.layers[1].get_weights()[1]
B_Output_Hidden = model.layers[3].get_weights()[1]
#print(summary)
print('INPUT-HIDDEN LAYER WEIGHTS:')
print(W_Input_Hidden)
print('HIDDEN-OUTPUT LAYER WEIGHTS:')
print(W_Output_Hidden)

print('INPUT-HIDDEN LAYER BIAS:')
print(B_Input_Hidden)
print('HIDDEN-OUTPUT LAYER BIAS:')
print(B_Output_Hidden)

# "Loss"
plt.plot(np.sqrt(history.history['loss']))
plt.plot(np.sqrt(history.history['val_loss']))
# plt.plot(history.history['v1_accuracy'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation','v1_accuracy'], loc='upper left')
plt.show()


# Para el problema de paridad
# NI=5
# x=[]
# for i in range(2**NI):
#     num=bin(i)[2:]                            # covierte decimal a binario
#     a=np.array([int(z) for z in str(num)])    # pone los numeros en un vector
#     while len(a)<NI:                          # llena con ceros si es necesario
#         a=np.insert(a,0,0)
#     x.append(a)
# x=np.array(x)
# y=np.zeros(2**NI)
# for i in range(2**NI):      # calculo salidas
#     y[i]=sum(x[i,:])%2
# x[x==0]=-1                  # opcional: representacion -1,1
# y[y==0]=-1                  # opcional: representacion -1,1
    