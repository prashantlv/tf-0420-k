import tensorflow as tf 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from urllib.request import urlretrieve

url = 'https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/moore.csv'
urlretrieve(url, 'moore.csv')
data = pd.read_csv('moore.csv', header=None).values
X = data[:,0].reshape(-1, 1) # make it a 2-D array of size N x D where D = 1
Y = data[:,1]

plt.scatter(X,Y) # exponential
Y = np.log(Y)
plt.scatter(X,Y) #better
X = X - X.mean()

model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(1,)),
  tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.SGD(0.001, 0.9), loss='mse')

# learning rate scheduler
def schedule(epoch, lr):
  if epoch >= 50:
    return 0.0001
  return 0.001
 

scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

r = model.fit(X, Y, epochs = 100, callbacks=[scheduler])

plt.plot(r.history['loss'], label='loss')
# Make sure the line fits our data
Yhat = model.predict(X).flatten()
plt.scatter(X, Y)
plt.plot(X, Yhat, color = 'r')

#perfect
