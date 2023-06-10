import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#for i in range(450):
#    imageio.imsave(f"images\\{i}.png", x_test[i])


plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
#plt.show()


x_train = x_train / 255
x_test = x_test / 255 

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        input_shape=(28, 28, 1), 
        filters = 32,
        kernel_size = (5, 5),
        padding='same', # дополнение краев нулями для свертки
        activation='relu'
    ),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(), #слой трансформции в новую размерность (массив превратит в одномерный)
    tf.keras.layers.Dense(512, activation=tf.nn.relu), #
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) #Параметры колличество нейронов и функция активации
])


"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu), 
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
"""

model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy']  
)

model.fit(x_train.reshape(-1, 28, 28, 1), y_train, epochs=5)

print(model.evaluate(x_test.reshape(-1, 28, 28, 1), y_test))

model.save('models\\digits1.h5')

#model = tf.keras.models.load_model('models\\digits1.h5')

def model_answer(model, filename, display=True):
    image = imageio.imread(filename, pilmode="RGB")
    image = np.mean(image, 2, dtype=float)
    image = image / 255
    if display:
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image, cmap=plt.cm.binary)
        plt.xlabel(filename)
        plt.show()

    image = np.expand_dims(image, 0)
    return np.argmax(model.predict(image))

for i in range(10):
    filename = f'digits\\{i}.png'
    print("Имя файла: ", filename, "\tОтвет: ", model_answer(model, filename, False))


print("Имя файла: ", 'digits\\7_1.png', "\tОтвет: ", model_answer(model, 'digits\\7_1.png', False))
print("Имя файла: ", 'digits\\7_2.png', "\tОтвет: ", model_answer(model, 'digits\\7_2.png', False))
print("Имя файла: ", 'digits\\7_3.png', "\tОтвет: ", model_answer(model, 'digits\\7_3.png', False))

print("Имя файла: ", 'digits\\0_1.png', "\tОтвет: ", model_answer(model, 'digits\\0_1.png', False))
print("Имя файла: ", 'digits\\2_1.png', "\tОтвет: ", model_answer(model, 'digits\\2_1.png', False))

print("Имя файла: ", 'digits\\4_1.png', "\tОтвет: ", model_answer(model, 'digits\\4_1.png', False))
print("Имя файла: ", 'digits\\4_2.png', "\tОтвет: ", model_answer(model, 'digits\\4_2.png', False))

print("Имя файла: ", 'digits\\8_1.png', "\tОтвет: ", model_answer(model, 'digits\\8_1.png', False))