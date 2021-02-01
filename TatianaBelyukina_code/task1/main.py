import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import sklearn

titanic = pd.read_csv('./train.csv')
titanic.head(10)

print(titanic.shape)
print(titanic.describe())
print(titanic['Survived'].value_counts())
sns.countplot(titanic['Survived'])
plt.show()
print(titanic.groupby('Sex')[['Survived']].mean())
print(titanic.pivot_table('Survived', index='Sex', columns='Pclass'))
titanic.pivot_table('Survived', index='Sex', columns='Pclass').plot()
plt.show()

age = pd.cut(titanic['Age'], [0,18,80])
print(titanic.pivot_table('Survived', ['Sex', age], 'Pclass'))

print(titanic.isna().sum())

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
    for i in range(len(diapasones)+1):
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


def preprocess(data):
    for i in range(data.shape[0]):
        if np.isnan(data["Age"][i]):
            if "Miss" in data["Name"][i] or "Ms" in data["Name"][i] :
                data["Age"][i] = 15
            elif "Mrs" in data["Name"][i] or "Mr" in data["Name"][i] :
                data["Age"][i] = 30
            elif "Don." in data["Name"][i] :
                data["Age"][i] = 40
            else :
                data["Age"][i] = 10

    data = data.drop(['PassengerId', 'Ticket', 'Name', 'Cabin'], axis=1)
    print(data)
    data = splitColumn(data, "Embarked")
    data = splitColumn(data, "Sex")
    data = splitColumn(data, "Pclass")
    data = splitFloatColumn(data, "Parch", [1,2,4])
    data = splitFloatColumn(data, "SibSp", [1,2,4])
    data = splitFloatColumn(data, "Age", [2,5,10,16,20,28,35,45])
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


titanic = preprocess(titanic)
print(titanic.dtypes)

X,Y = split_data(titanic)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


def check_model(x_train, y_train, x_test, y_test, model):
    print("traindata:", model.score(x_train, y_train))
    print("testdata:" , model.score(x_test, y_test))
    print()


def models(x_train, y_train, x_test, y_test):
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion="entropy")
    tree.fit(x_train,y_train)

    from sklearn.linear_model import LogisticRegression
    logReg = LogisticRegression(max_iter=1000000)
    logReg.fit(x_train,y_train)

    from sklearn.neighbors import KNeighborsClassifier
    neighbors = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
    neighbors.fit(x_train,y_train)

    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=10, criterion="entropy")
    forest.fit(x_train,y_train)

    from sklearn.naive_bayes import GaussianNB
    gauss = GaussianNB()
    gauss.fit(x_train,y_train)

    from sklearn.svm import SVC
    svc = SVC(kernel="linear")
    svc.fit(x_train,y_train)
    
    svcRBF = SVC(kernel="rbf")
    svcRBF.fit(x_train,y_train)

    print("tree")
    check_model(x_train, y_train, x_test, y_test, tree)
    print("logReg")
    check_model(x_train, y_train, x_test, y_test, logReg)
    print("neighbors")
    check_model(x_train, y_train, x_test, y_test, neighbors)
    print("forest")
    check_model(x_train, y_train, x_test, y_test, forest)
    print("gauss")
    check_model(x_train, y_train, x_test, y_test, gauss)
    print("svc_linear")
    check_model(x_train, y_train, x_test, y_test, svc)
    print("svc_rbf")
    check_model(x_train, y_train, x_test, y_test, svcRBF)
    return svcRBF


model = models(X_train, Y_train, X_test, Y_test)


def generateAns(model):
    test_ids = pd.read_csv("test.csv")["PassengerId"]
    test = pd.read_csv("test.csv")

    test = preprocess(test)
    x, y = split_data(test)

    Y_pred = model.predict(x)
    print(Y_pred)
    Y_pred = Y_pred.astype(int) 
    print(Y_pred)

    submission = pd.DataFrame({
            "PassengerId": test_ids,
            "Survived": Y_pred
        })
    submission.to_csv('./titanic.csv', index=False)
    print('Exported!')

generateAns(model)