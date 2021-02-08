import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
import pandas as pd

from google.colab import drive
drive.mount('path')

input_path = 'path'
output_path = 'path'


IMG_SIZE = (160, 160)
BATCH_SIZE = 32
epochs = 10
tuning_epochs = 8
learning_rate = 0.0001
validation_koef = 5

train_dataset = image_dataset_from_directory(input_path,
                                             shuffle=True,
                                             subset='training',
                                             seed=1,
                                             validation_split=0.2,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

validation_dataset = image_dataset_from_directory(input_path,
                                                  shuffle=True,
                                                  subset='validation',
                                                  seed=1,
                                                  validation_split=0.2,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)

validation_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(validation_batches//validation_koef)
validation_dataset = validation_dataset.skip(
    validation_batches//validation_koef)

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_change = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip(mode='horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(.5, .2),
    tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.1),
    tf.keras.layers.experimental.preprocessing.RandomTranslation(
        height_factor=0.1, width_factor=0.1)
])

base_model = tf.keras.applications.EfficientNetB0(input_shape=IMG_SIZE+(3,),
                                                  include_top=False,
                                                  weights='imagenet',
                                                  drop_connect_rate=0.4)
base_model.trainable = False
model = tf.keras.Sequential()
model.add(base_model)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(1))

print(model.summary())

input = tf.keras.Input(IMG_SIZE+(3,))
x = base_model(data_change(input), training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.BatchNormalization()(x)
output = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(input, output)

print(model.summary())

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_dataset,
                    epochs=epochs,
                    validation_data=validation_dataset)

wqe = model.evaluate(test_dataset)
print(wqe)
model.save('path')
print("Done!")
