import tensorflow as tf


def chooseBestModel(x_train, y_train, x_test, y_test):
    activations = ['relu', 'sigmoid', 'softmax']
    f = open("rez.txt", "a")

    for i in range(10, 150):
        for ai in activations:
            for j in range(10, 150):
                for aj in activations:
                    f.write(str([[i, ai], [j, aj]]) + " " + str(testLData(x_train, y_train, x_test, y_test, [[i, ai], [j, aj]])) + "\n")
    f.close()


def ann_predict(model, x, possible_y):
    y = model.predict(x)
    y = (y[:, 0] >= 0.5).astype(int) 
    return y


def testLData(x_train, y_train, x_test, y_test, lData):
    s = 0
    for i in range(10):
        s += annModel(x_train, y_train, x_test, y_test, lData)[1]
    return s / 10


def annModel(x_train, y_train, x_test, y_test, lData):
    layers = [tf.keras.layers.Flatten(input_shape=(33,))]
    for i in lData:
        layers.append(tf.keras.layers.Dense(i[0], activation=i[1]))
    layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))

    model = tf.keras.Sequential(layers)
    y_train = y_train.astype(float)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.fit(x_train, y_train, epochs=100)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("accuracy:", test_acc)
    return model, test_acc
