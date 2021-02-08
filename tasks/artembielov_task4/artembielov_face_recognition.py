import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow import keras
import kerastuner as kt

import IPython


model_name = 'model_v3.h5'
path = "..."

activation_function = 'relu'
batch_size = 128
img_height = 224
img_width = 224

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

class_names = ds_train.class_names

feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor_layer = hub.KerasLayer(
    feature_extractor_model, input_shape=(img_height, img_width, 3), trainable=False)


def model_create(hp):
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    num_classes = len(class_names)
    model = Sequential([
        feature_extractor_layer,
        tf.keras.layers.Dense(num_classes),
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

tuner.search(ds_train, epochs=10, validation_data=ds_val, callbacks=[ClearTrainingOutput()])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

model = tuner.hypermodel.build(best_hps)
model.fit(ds_train, epochs=10, validation_data=ds_val)

model.save('models/' + model_name)
