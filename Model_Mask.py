import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import os
import keras as k
from keras.optimizers import Adam
from tensorflow.keras import datasets, layers, models
from keras.models import load_model, Model
import pickle
import cv2
from PIL import Image

# untuk memaksimalkan penggunaan GPU
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# dir = "C:\\Users\\wijay\\Desktop\\Prototype 2\\dataset"

# categories = ["with_mask", "without_mask"]

# for category in categories:
#     print(category)
#     path = os.path.join(dir, category)
#     for img in os.listdir(path):
#     	img_path = os.path.join(path, img)
#     	image = load_img(img_path, target_size=(150, 150))
#     	image = img_to_array(image)
#     	image = image / 255.0

#     	data.append(image)
#     	labels.append(category)


data = []
labels = []
labels_one_dim = []

# saving data and lable list
file_data_name = "data.pkl"
file_lable_name = "lable.pkl"

# open_file_data = open(file_data_name, "wb")
# pickle.dump(data, open_file_data)
# open_file_data.close()

# open_file_label = open(file_lable_name, "wb")
# pickle.dump(labels, open_file_label)
# open_file_label.close()


open_file_data = open(file_data_name, "rb")
data = pickle.load(open_file_data)
open_file_data.close()

open_file_label = open(file_lable_name, "rb")
labels = pickle.load(open_file_label)
open_file_label.close()

one_encode = OneHotEncoder(sparse=False)
labels = np.reshape(labels, (len(labels), 1))
labels = one_encode.fit_transform(labels)

for i in range(len(labels)):
    if labels[i][0] == 1:
        val = 1
        labels_one_dim.append(val)
    elif labels[i][1] == 1:
        val = 0
        labels_one_dim.append(val)


data = np.array(data, dtype='float32')
labels = np.array(labels_one_dim)

(train_images, test_images, train_labels, test_labels) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

datagen = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest",
	)

train_generator = datagen.flow(train_images, train_labels, batch_size=16)

num_filters=64
epoch = 100
ac='relu'
adm=Adam(lr=0.0001, epsilon=1e-7)
drop_dense=0.5
drop_conv=0.2

model = models.Sequential()
model.add(layers.Conv2D(num_filters, (3, 3),activation=ac,input_shape=(150, 150, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(num_filters, (3, 3),activation=ac))
model.add(layers.BatchNormalization())
model.add(layers.AveragePooling2D(pool_size= (2,2)))
model.add(layers.Dropout(drop_conv))

model.add(layers.Conv2D(2*num_filters, (3, 3),activation=ac))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(2*num_filters, (3, 3),activation=ac))
model.add(layers.BatchNormalization())
model.add(layers.AveragePooling2D(pool_size= (2,2)))
model.add(layers.Dropout(2 * drop_conv))

model.add(layers.Conv2D(4*num_filters, (3, 3), activation=ac))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(4*num_filters, (3, 3),activation=ac))
model.add(layers.BatchNormalization())
model.add(layers.AveragePooling2D(pool_size= (2,2)))
model.add(layers.Dropout(3 * drop_conv))

model.add(layers.Conv2D(8*num_filters, (3, 3), activation=ac))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(8*num_filters, (3, 3),activation=ac))
model.add(layers.BatchNormalization())
model.add(layers.AveragePooling2D(pool_size= (2,2)))
model.add(layers.Dropout(4 * drop_conv))


model.add(layers.Flatten())
model.add(layers.Dense(125, activation=ac))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(drop_dense))
model.add(layers.Dense(50, activation=ac))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(drop_dense))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=adm)

mc = ModelCheckpoint("model_checkpoint\\weights-improvement-{epoch:02d}.h5", monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
history=model.fit_generator(train_generator, steps_per_epoch = len(train_images) //16, 
							epochs=epoch, validation_data=(test_images, test_labels), callbacks=[mc])

loss,accuracy = model.evaluate(test_images, test_labels)
print("Accuracy is : ",accuracy)
print("Loss is : ",loss)

# plot training akurasi dan loss
N = epoch
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.show()


