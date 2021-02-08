import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow import keras
import kerastuner as kt

import IPython

path = "..."

activation_function = 'relu'
batch_size = 128
img_height = 180
img_width = 180

ds_train = tf.keras.preprocessing.image_dataset_from_directory(path, labels='inferred', color_mode='rgb',
                                                               batch_size=batch_size,
                                                               image_size=(img_height, img_width),
                                                               shuffle=True, seed=123,
                                                               validation_split=0.2,
                                                               subset='training')

ds_val = tf.keras.preprocessing.image_dataset_from_directory(path, labels='inferred', color_mode='rgb',
                                                             batch_size=batch_size,
                                                             image_size=(img_height, img_width),
                                                             shuffle=True,
                                                             seed=123,
                                                             validation_split=0.2,
                                                             subset='validation')


def normalize(image, label):
    image = tf.cast(image / 255., tf.float32)
    return image, label


class_names = ds_train.class_names
ds_train = ds_train.map(normalize)
ds_val = ds_val.map(normalize)

AUTOTUNE = tf.data.AUTOTUNE

ds_train = ds_train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
ds_val = ds_val.cache().prefetch(buffer_size=AUTOTUNE)


def model_create(hp):
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    num_classes = len(class_names)
    model = Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation=activation_function),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation=activation_function),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation=activation_function),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(units=hp_units, activation=activation_function),
        layers.Dense(num_classes)
    ])

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)


tuner = kt.Hyperband(model_create,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

tuner.search(ds_train,
             epochs=10,
             validation_data=ds_val,
             callbacks=[ClearTrainingOutput()])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

model = tuner.hypermodel.build(best_hps)
model.fit(ds_train,
          epochs=10,
          validation_data=ds_val)
