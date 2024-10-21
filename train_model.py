import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.src.models.sequential import Sequential
from keras.src.layers.convolutional.conv2d import Conv2D
from keras.src.layers.core.dense import Dense
from keras.src.layers.reshaping.flatten import Flatten
from keras.src.layers.regularization.dropout import Dropout
from keras.src.layers.pooling.max_pooling2d import MaxPooling2D as MaxPool2D
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers.adam import Adam

import tensorflow as tf

import cv2
import os

import numpy as np

# Tentukan label dan ukuran gambar
labels = ['Covid', 'Normal']
img_size = 224

def get_data(data_dir):
    images = []
    labels_list = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                resized_arr = np.expand_dims(resized_arr, axis=-1)
                images.append(resized_arr)
                labels_list.append(class_num)
            except Exception as e:
                print(e)
    return np.array(images), np.array(labels_list)

# Ganti dengan path dataset Anda
train_images, train_labels = get_data(r'D:\Kecerdasan Buatan\dataset\train')
val_images, val_labels = get_data(r'D:\Kecerdasan Buatan\dataset\validation')

l = []
for i in range (len(train_labels)):
    if(train_labels[i] == 0):
        l.append("Covid")
    else :
        l.append("Normal")
sns.set_style('darkgrid')
sns.countplot(l)

x_train = []
y_train = []
x_val = []
y_val = []

for image in train_images:
  x_train.append(image)

for label in train_labels:
  y_train.append(label)

for image in val_images:
  x_val.append(image)

for label in val_labels:
  y_val.append(label)

x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

datagen = ImageDataGenerator(
        featurewise_std_normalization=True,  # bagi dengan standar deviasi dataset
        samplewise_std_normalization=True,  # bagi dengan standar deviasi input itu sendiri
        rotation_range = 90,  # rotasi gambar 0 sampai 90 derajat
        zoom_range = 0.2, # zoom-in dan zoom-out
        width_shift_range=0.1,  # geser horizontal
        height_shift_range=0.1,  # geser vertikal
        horizontal_flip = True,  # flip secara horizontal
        vertical_flip= True)  # flip secara vertikal

datagen.fit(x_train)

# Buat model
model = Sequential()
model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(img_size, img_size, 1)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="softmax"))

# Tentukan optimizer dan loss function
optimisasi = Adam(learning_rate=0.00001)
model.compile(optimizer=optimisasi, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Latih model
history = model.fit(x_train, y_train, epochs=25, validation_data=(x_val, y_val))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.xticks(range(min(epochs_range), max(epochs_range)+1, 5))
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.xticks(range(min(epochs_range), max(epochs_range)+1, 5))
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.savefig('plot.png')
plt.show()

# Simpan model sebagai model.h5
model.save('model.h5')