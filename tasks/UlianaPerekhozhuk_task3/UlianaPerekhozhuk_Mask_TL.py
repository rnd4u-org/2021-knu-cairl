import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


path = "..."

activation_function = 'relu'

batch_size = 64
img_height = 224
img_width = 224


ds_train = tf.keras.preprocessing.image_dataset_from_directory(path, labels='inferred', color_mode='rgb',
                                                               batch_size=batch_size,
                                                               image_size=(img_height, img_width),
                                                               validation_split=0.2,
                                                               subset='training',
                                                               shuffle=True, seed=123)

ds_val = tf.keras.preprocessing.image_dataset_from_directory(path, labels='inferred', color_mode='rgb',
                                                             batch_size=batch_size,
                                                             image_size=(img_height, img_width),
                                                             validation_split=0.2,
                                                             subset='validation',
                                                             shuffle=True,
                                                             seed=123)

class_names = ds_train.class_names
num_classes = len(class_names)

feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor_layer = hub.KerasLayer(
    feature_extractor_model, input_shape=(img_height, img_width, 3), trainable=False)


model = Sequential([
          feature_extractor_layer,
          layers.Dense(num_classes)
        ])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(ds_train, validation_data=ds_val, epochs=10)
