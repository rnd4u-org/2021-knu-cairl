from sklearn_models import sklearn_models, boosted_test
from ann_model import annModel, ann_predict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

data = pd.read_csv("train.csv")
print(data.shape)

def splitColumn(data, feature):
    vals = set(data[feature])
    for val in vals:
        if str(val) == "nan":
            continue
        a = []
        for i in range(data.shape[0]):
            if data[feature][i] == val:
                a.append(1)
            else: a.append(0)
        data[feature + "_" + str(val)] = a
    return data.drop([feature], axis=1)


def splitFloatColumn(data, feature, diapasones):
    for i in range(len(diapasones) + 1):
        a = []
        for j in range(data.shape[0]):
            t = True
            if i > 0:
                t &= (data[feature][j] >= diapasones[i - 1])
            if i < len(diapasones):
                t &= (data[feature][j] < diapasones[i])
            if t:
                a.append(1)
            else: a.append(0)
        data[feature + "_" + str(i)] = a
    return data.drop([feature], axis=1)


def showSurvival(data):
    features = ['Fare']
    rows = 1
    cols = 1

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))

    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            sns.countplot(data[features[i]], hue=data["Survived"], ax=axs)
            axs.legend(title="Survived", loc='upper right')

    plt.tight_layout()
    plt.show()


def printVals(data):
    for val in data:
        print(data[val].value_counts())
        print()


def preprocess(data):
    for i in range(data.shape[0]):
        if np.isnan(data["Age"][i]):
            # data["Age"][i] = 1000
            if "Ms" in data["Name"][i] or "Miss" in data["Name"][i]:
                data["Age"][i] = 15
            elif "Mr" in data["Name"][i] or "Mrs" in data["Name"][i]:
                data["Age"][i] = 30
            elif "Don." in data["Name"][i]:
                data["Age"][i] = 40
            else:
                data["Age"][i] = 10

    data = data.drop(['PassengerId', 'Ticket', 'Name', 'Cabin'], axis=1)
    print(data)
    data = splitColumn(data, "Embarked")
    data = splitColumn(data, "Sex")
    data = splitColumn(data, "Pclass")
    data = splitFloatColumn(data, "Parch", [1, 2, 4])
    data = splitFloatColumn(data, "SibSp", [1, 2, 4])
    data = splitFloatColumn(data, "Age", [2, 5, 10, 16, 20, 28, 35, 45])
    data = splitFloatColumn(data, "Fare", [7, 9, 10, 15, 30, 60, 80])

    for feature in data:
        print(feature)
        for i in range(data.shape[0]):
            if np.isnan(data[feature][i]):
                data[feature][i] = 0

    print(data)
    return data


def split_data(data):
    return (data.iloc[:, 1:].values, data.iloc[:, 0].values) 


data = preprocess(data)
x, y = split_data(data)

a = []
for i in range(1):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train = x
    y_train = y
    # exit()
    ann, acc = annModel(x_train, y_train, x_test, y_test, [[50, 'relu']])

    # chooseBestModel(x_train, y_train, x_test, y_test)
    models = sklearn_models(x_train, y_train, x_test, y_test)

    rez = boosted_test(models, x_test, y_test)
    a.append(rez)

    print()
    print(a)
    print("mean:", sum(a) / len(a))


def generateAns():
    test_ids = pd.read_csv("test.csv")["PassengerId"]
    test = pd.read_csv("test.csv")

    test = preprocess(test)
    x, y = split_data(test)

    # Y_pred = models_predict(models, x)
    y = ann_predict(ann, x, [])

    submission = pd.DataFrame({
        "PassengerId": test_ids,
        "Survived": y
    })
    submission.to_csv('./titanic.csv', index=False)
    print('Exported!')


generateAns()