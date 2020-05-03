
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
print('data.shape :', data.data.shape)
print('target_names :', data.target_names)

x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.25)
N, D = x_train.shape

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(D,)),
        tf.keras.layers.Dense(1,activation='sigmoid')      # 1 is output here
                              ])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100)

plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()

plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()

P = model.predict(x_test)
#print(P)

import numpy as np
P = np.round(P).flatten()
print(P)

ev = model.evaluate(x_test, y_test)
print('accuracy :'ev)
