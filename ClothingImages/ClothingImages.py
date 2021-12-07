import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

images=tf.keras.datasets.fashion_mnist
(train_image, train_labels),(test_image, test_labels)=images.load_data()
train_images = train_image / 255.0
test_images = test_image / 255.0

#print(train_images.shape)
#60000 images (28x28 pixels)
#plt.figure()
#plt.imshow(train_images[0])
#plt.show()

model=tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(100, activation='sigmoid'))
model.add(tf.keras.layers.Dense(10))

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
#categorical_crossentropy ( cce ) produces a one-hot array containing the probable match for each category(label).(label)
#sparse_categorical_crossentropy ( scce ) produces a category index of the most likely matching category.

model.fit(train_images, train_labels,epochs=10)
model.evaluate(test_images,test_labels)
