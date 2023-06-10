import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Activation, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

from tensorflow.keras.utils import plot_model
from IPython.display import display, Image

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


x = np.arange(-10, 10, 0.2)
y = np.sin(x) 

#plt.plot(x, y, 'bo')
#plt.show()


def my_plot(model):
  plot_model(
      model,
      to_file="model.png",
      show_shapes=True,
      show_layer_names=True,
      rankdir="TB",
      #expand_nested=False,
      #dpi=96
  )
  image = Image('model.png')
  display(image)


input = tf.keras.layers.Input(shape=(1),name = 'Input')
a = tf.keras.layers.Dense(25, activation='relu')(input)
a = tf.keras.layers.Dense(10, activation='relu')(a)
a = tf.keras.layers.Dense(1, activation=None)(a)

model = Model(input, a)

my_plot(model)

#optimizer = SGD(learning_rate=0.0001)
#optimizer = tf.keras.optimizers.RMSprop(0.001)
optimizer = Adam(learning_rate=0.0003, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile( optimizer=optimizer ,
               loss = 'mse'
             )

EPOCHS = 8000

model.fit( x, y,
    epochs=EPOCHS,
    batch_size=32,
)

print("Model trained")

#model.save('models\\sinus.h5')


#model = tf.keras.models.load_model('models\\sinus.h5')

x1 = np.arange(-15, 15, 0.2)
pred = model.predict(x1)
#print(pred)
plt.plot(x, y, 'bo', x1, pred, 'r')
plt.show()

x1 = np.arange(-9, 9, 0.2)
pred = model.predict(x1)
#print(pred)
plt.plot(x, y, 'bo', x1, pred, 'r')
plt.show()

aboba = 0
