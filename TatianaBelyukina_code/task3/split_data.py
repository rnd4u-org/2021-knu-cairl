import numpy as np 
import os
import cv2
from random import shuffle


def create_data():
    test_size = 0.2
    inputs = []
    outputs = []

    for f in os.listdir('with_mask'):
        outputs.append(np.array([1, 0]))
        image = cv2.resize(cv2.imread('with_mask/' + f), (224, 224))
        inputs.append(np.array(image))
    print("with_mask finish")

    for f in os.listdir('without_mask'):    
        outputs.append(np.array([0, 1]))
        image = cv2.resize(cv2.imread('without_mask/' + f), (224, 224))
        inputs.append(np.array(image))

    print("without_mask finish")
    print(len(inputs))

    data = list(zip(inputs, outputs))
    shuffle(data)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for image in data[:int(test_size*len(data))]:
        x_test.append(image[0])
        y_test.append(image[1])

    for image in data[int(test_size * len(data)):]:
        x_train.append(image[0])
        y_train.append(image[1])

    np.save("inputs_test.npy", x_test)
    np.save("inputs_train.npy", x_train)
    np.save("outputs_test.npy", y_test)
    np.save("outputs_train.npy", y_train)


def load_data():
    return (np.load("inputs_train.npy"), np.load("outputs_train.npy"), np.load("inputs_test.npy"), np.load("outputs_test.npy"))