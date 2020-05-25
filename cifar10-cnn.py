import tensorflow as tf 
from tensorflow.keras import layers, models. datasets
import matplotlib.lyplot as plt 

cifar10 = datasets.cifar10
(x_train, y_train),(x_test, y_test) = cifar10.load_data()
x_train, x_test = x-train/255.0, x_test/255.0

print('x_train.shape :',x_train.shape)  #(50000, 32, 32, 3)
print('y_train.shape :',y_train.shape)  #(50000, 1)
print('t_test.shape :',y_test.shape)    #(50000, 1)

y_train, y_test = y_train.flatten(), y_test.flatten()
y_test.shape    #(50000,)

k = len(set(y_test))    # 10 unique features

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
r = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=10)       

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2) 


plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()