import tensorflow as tf
from load_data import load_data
from model import build_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD

def train():
    x_train, x_test, y_train, y_test = load_data()
    # print(x_train.shape)
    # print(x_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)
    # exit()
    model = build_model()
    size = len(model.layers)
    steps = 3
    batch = 10
    epochs_per_train = 10

    earlystop = EarlyStopping(patience=3, monitor='accuracy')
    learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.0001)
    callbacks = [earlystop, learning_rate_reduction]

    for step in range(steps - 1):
        for i in range(size):
            model.layers[i].trainable = size - size / steps * step > i and size - size / steps * (step + 1) >= i
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
            history = model.fit(
                x=x_train, 
                y=y_train,
                epochs=epochs_per_train,
                validation_data=(x_test, y_test),
                validation_steps=len(x_test) // batch,
                steps_per_epoch=len(x_train) // batch,
                callbacks=callbacks)
            model.save("model" + str(step))

    for i in range(size):
        model.layers[i].trainable = size - size / steps * (steps - 1) > i

    model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        x=x_train, 
        y=y_train,
        epochs=epochs_per_train,
        validation_data=(x_test, y_test),
        validation_steps=len(x_test) // batch,
        steps_per_epoch=len(x_train) // batch,
        callbacks=callbacks)

    model.save("final_model")
    print(history)
    print(model.summary())

def evaluate(name):
    model = tf.keras.models.load_model(name)
    print(len(model.layers))
    x_train, x_test, y_train, y_test = load_data()
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test)
    print("test loss, test acc:", results)

if __name__ == "__main__":
    train()