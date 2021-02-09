import numpy as np
import os
import cv2
from random import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm 


def lable_img(name):
    if "out" in name:
        return np.array([0, 1])
    else:
        return np.array([1, 0])


def load_data():
    dirs = ["./data/with_mask", "./data/without_mask"]
    size = 224

    try:
        x_train = np.load('x_train.npy', allow_pickle=True)
        x_test = np.load('x_test.npy', allow_pickle=True)
        y_train = np.load('y_train.npy', allow_pickle=True)
        y_test = np.load('y_test.npy', allow_pickle=True)

        return (x_train, x_test, y_train, y_test)
    except FileNotFoundError:
        pass

    indexes = []
    images = []
    lables = []
    ind = 0
    for d in dirs:
        for img in tqdm(os.listdir(d)):
            lable = lable_img(img)
            path = os.path.join(d, img)
            img = cv2.resize(cv2.imread(path), (size, size))
            images.append(np.array(img))
            lables.append(lable)
            indexes.append(ind)
            ind += 1

    shuffle(indexes)
    n = len(images) * 4 // 5

    x_train, x_test, y_train, y_test = train_test_split(images, lables, test_size=0.2)

    np.save('x_train.npy', x_train)
    np.save('x_test.npy', x_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

    return load_data()